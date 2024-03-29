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
from ..base_blocks import SelfAttnBlockApi, CrossAttnBlockApi, TransformerEncoder
from ..base_blocks import TransformerDecoder, PointTransformerEncoder
from ..base_blocks import PointTransformerDecoder, PointTransformerEncoderEntry
from ..base_blocks import PointTransformerDecoderEntry, DGCNN_Grouper, Encoder
from ..base_blocks import SimpleEncoder, Fold, SimpleRebuildFCLayer
from ..base_blocks import CycleLR
from ..image_encoders import ResNet18
from .pctransformer import PCTransformer


class PCTransformerImgOnly(PCTransformer):
    def __init__(self, config):
        # Adjust config to enable image features
        config.use_img_features = True
        config.use_seg_features = False  # Ensure segmentation features are disabled
        config.enable_denoising = True
        config.enable_segmentation_enhancement = False
        config.seg_before_enc = False
        config.seg_before_dec = False
        super().__init__(config)
        

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
        self.base_model = PCTransformerImgOnly(config)
        
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
        # print('self.alpha_loss:', self.alpha_loss)

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