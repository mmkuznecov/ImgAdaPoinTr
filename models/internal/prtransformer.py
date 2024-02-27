import time
from functools import partial, reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from pytorch3d.ops import points_normals
from timm.models.layers import DropPath, trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1
from ..build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from ..segmentation.GDANet_ptseg import GDANet
from ..base_blocks import SelfAttnBlockApi, CrossAttnBlockApi, TransformerEncoder
from ..base_blocks import TransformerDecoder, PointTransformerEncoder
from ..base_blocks import PointTransformerDecoder, PointTransformerEncoderEntry
from ..base_blocks import PointTransformerDecoderEntry, DGCNN_Grouper, Encoder
from ..base_blocks import SimpleEncoder, Fold, SimpleRebuildFCLayer
from ..base_blocks import CycleLR
from ..image_encoders import ResNet18


class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim)
        )
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)

        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Additional features based on config
        if hasattr(config, 'use_img_features') and config.use_img_features:
            self.im_encoder = ResNet18()
            self.img_dim = 384
            self.get_better_img_size = nn.Sequential(
                nn.Linear(196, self.img_dim),
                nn.GELU()
            )
            self.init_img_attention_layers()

        if hasattr(config, 'use_seg_features') and config.use_seg_features:
            self.segmentator = GDANet(50)
            self.img_dim = 384  # Assuming img_dim is the same for both img and seg features for simplicity
            self.get_better_seg_size = nn.Sequential(
                nn.Linear(128, self.img_dim),
                nn.GELU()
            )
            self.init_seg_attention_layers()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_img_attention_layers(self):
        self.cross_attn_img = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm_img = nn.LayerNorm(self.img_dim)
        self.self_attn_img = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm_img2 = nn.LayerNorm(self.img_dim)

    def init_seg_attention_layers(self):
        self.cross_attn_seg = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm_seg = nn.LayerNorm(self.img_dim)
        self.self_attn_seg = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm_seg2 = nn.LayerNorm(self.img_dim)
        
    def forward(self, xyz, img=None, cls_vec=None):

        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor)  # b n c

        # Process image features if enabled
        if hasattr(self.config, 'use_img_features') and self.config.use_img_features:
            img_feat = self.im_encoder(img)
            img_feat = self.get_better_img_size(img_feat)
            img_feat = img_feat.transpose(0, 1)
            x = x.transpose(0, 1)

            # Image feature attention
            x, _ = self.cross_attn_img(x, img_feat, img_feat)
            x = self.layer_norm_img(x)
            x, _ = self.self_attn_img(x, x, x)
            x = self.layer_norm_img2(x + x)
            x = x.transpose(0, 1)

        # Process segmentation features if enabled
        if hasattr(self.config, 'use_seg_features') and self.config.use_seg_features and cls_vec is not None:
            norm_plt = points_normals.estimate_pointcloud_normals(coor, 30, disambiguate_directions=False)
            seg_emb, seg_idx = self.segmentator(coor.transpose(1, 2), norm_plt, cls_vec)
            seg_emb = seg_emb.transpose(1, 2).transpose(0, 1)
            seg_emb = self.get_better_seg_size(seg_emb)
            x = x.transpose(0, 1)

            # Segmentation feature attention
            x, _ = self.cross_attn_seg(x, seg_emb, seg_emb)
            x = self.layer_norm_seg(x)
            x, _ = self.self_attn_seg(x, x, x)
            x = self.layer_norm_seg2(x + x)
            x = x.transpose(0, 1)

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        coarse_inp = misc.fps(xyz, self.num_query // 2)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 224+128 3?

        mem = self.mem_link(x)

        # Query selection
        query_ranking = self.query_ranking(coarse)  # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
        coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))

        # Produce query
        q = self.mlp_query(torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1), coarse], dim=-1))  # b n c

        # Forward decoder
        q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

        if self.training:
            
            if self.config.enable_denoising:
                picked_points = misc.fps(xyz, 64)
                picked_points = misc.jitter_points(picked_points)
                size_coarse_wo_denoise = coarse.shape[1]
                coarse = torch.cat([coarse, picked_points], dim=1)  # B 256+64 3?
                denoise_length = 64

                q = self.mlp_query(torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1), coarse], dim=-1))

                q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            if self.config.enable_segmentation_enhancement:
                norm_plt = points_normals.estimate_pointcloud_normals(coarse[:, :size_coarse_wo_denoise, :], 30, disambiguate_directions=False)
                seg_emb, seg_idx = self.segmentator(coarse[:, :size_coarse_wo_denoise, :].transpose(1, 2), norm_plt, cls_vec)
                seg_emb = seg_emb.transpose(1, 2).transpose(0, 1)
                seg_emb = self.get_better_seg_size2(seg_emb)

                q = q.transpose(0, 1)

                q_temp = q.clone()
                q_temp = q_temp[:size_coarse_wo_denoise]

                q_out, _ = self.cross_attn4(q_temp, seg_emb, seg_emb)
                q_temp = self.layer_norm6(q_out + q_temp)

                q_out, _ = self.self_attn3(q_temp, q_temp, q_temp)
                q_temp = self.layer_norm7(q_out + q_temp)
                q_skip = q_temp

                q_out, _ = self.cross_attn5(q_temp, seg_emb, seg_emb)
                q_temp = self.layer_norm8(q_out + q_temp)

                q_out, _ = self.self_attn4(q_temp, q_temp, q_temp)
                q_temp = self.layer_norm9(q_out + q_temp)

                q_out, _ = self.cross_attn6(q_temp, q_skip, q_skip)
                q_temp = self.layer_norm10(q_out + q_temp)

                q[:size_coarse_wo_denoise] = q_temp
                q = q.transpose(0, 1)

            return q, coarse, denoise_length if self.config.enable_denoising else 0 
                

        return q, coarse, None


class PCTransformerImgOnly(PCTransformer):
    def __init__(self, config):
        # Adjust config to enable image features
        config.use_img_features = True
        config.use_seg_features = False  # Ensure segmentation features are disabled
        config.enable_denoising = True
        config.enable_segmentation_enhancement = False
        super().__init__(config)

class PCTransformerSegOnly(PCTransformer):
    def __init__(self, config):
        # Adjust config to enable segmentation features
        config.use_img_features = False  # Ensure image features are disabled
        config.use_seg_features = True
        config.enable_denoising = True
        config.enable_segmentation_enhancement = True
        super().__init__(config)

class PCTransformerImgSeg(PCTransformer):
    def __init__(self, config):
        # Adjust config to enable both image and segmentation features
        config.use_img_features = True
        config.use_seg_features = True
        config.enable_denoising = True
        config.enable_segmentation_enhancement = False
        super().__init__(config)
