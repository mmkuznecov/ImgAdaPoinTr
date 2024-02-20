##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################

from functools import partial, reduce

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from torchvision import transforms,models
import numpy as np

from extensions.chamfer_dist import ChamferDistanceL1
from ..build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from base_blocks import SelfAttnBlockApi, CrossAttnBlockApi, TransformerEncoder
from base_blocks import TransformerDecoder, PointTransformerEncoder
from base_blocks import PointTransformerDecoder, PointTransformerEncoderEntry
from base_blocks import PointTransformerDecoderEntry, DGCNN_Grouper, Encoder
from base_blocks import SimpleEncoder, Fold, SimpleRebuildFCLayer
from base_blocks import CycleLR
from image_encoders import ResNet18



######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
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
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
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
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.im_encoder = ResNet18()
        self.img_dim = 384
        self.get_better_size = nn.Sequential(
            nn.Linear(196, self.img_dim),
            nn.GELU()
        )
        self.cross_attn1 = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm1 = nn.LayerNorm(self.img_dim)

        self.self_attn1 = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm2 = nn.LayerNorm(self.img_dim)
        
        self.cross_attn2 = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm3 = nn.LayerNorm(self.img_dim)
        
        self.self_attn2 = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm4 = nn.LayerNorm(self.img_dim)
        
        
        self.cross_attn3 = nn.MultiheadAttention(self.img_dim, 8)
        self.layer_norm5 = nn.LayerNorm(self.img_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, img):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor) # b n c
        
        #add Img
        img_feat = self.im_encoder(img)
        img_feat = self.get_better_size(img_feat)
        img_feat = img_feat.transpose(0,1)
        x = x.transpose(0,1)
        
        # layer 1: cross + self attention
        x_out, _ = self.cross_attn1(x , img_feat, img_feat)
        x = self.layer_norm1(x_out + x) # b n c
        
        x_out, _ = self.self_attn1(x, x, x)
        x = self.layer_norm2(x_out + x)
        pc_skip = x
        
        # layer 2: cross + self attention
        x_out, _ = self.cross_attn2(x , img_feat, img_feat)
        x = self.layer_norm3(x_out + x) # b n c
        
        x_out, _ = self.self_attn2(x, x, x)
        x = self.layer_norm4(x_out + x)
        
        x_out, _ = self.cross_attn3(x, pc_skip, pc_skip)
        x = self.layer_norm5(x_out + x)
        x = x.transpose(0,1)
        #end img block
        
        
        global_feature = self.increase_dim(x) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            size_coarse_wo_denoise = coarse.shape[1]
#             print(size_coarse_wo_denoise)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)
            
            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0
        

######################################## PoinTr ########################################  

STEP_SIZE = 5
scheduler_loss = CycleLR(step_size=STEP_SIZE, max_lr=1.0, base_lr=0.01, gamma=0.995)
    

@MODELS.register_module()
class ImgResNetEncAdaPoinTrVariableLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = PCTransformer(config)
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step**2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step**2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()
        self.alpha_loss = [scheduler_loss.get_lr(last_epoch=epoch) for epoch in range(STEP_SIZE, 600)]
        print('self.alpha_loss:', self.alpha_loss)

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        
        assert pred_fine.size(1) == gt.size(1)
        
        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse  * self.alpha_loss[epoch] + loss_fine 

        return loss_denoised, loss_recon

    def forward(self, xyz, img):
        q, coarse_point_cloud, denoise_length = self.base_model(xyz, img) # B M C and B M 3
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        
        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret
        