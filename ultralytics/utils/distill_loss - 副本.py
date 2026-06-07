import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.checks import check_version
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.torch_utils import de_parallel

import functools
import numpy as np


class RTDETRLogicLoss(nn.Module):
    def __init__(self, hyp):
        super().__init__()

        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def power_transform(self, array, power=2):
        return torch.where(array < 0.5, array ** power, array ** (1/power))
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox_iou = torch.zeros(1, device=self.device)  # box iou loss
        lbox_l1 = torch.zeros(1, device=self.device)  # box l1 loss
        
        s_dec_bboxes, s_dec_scores, s_enc_bboxes, s_enc_scores, s_dn_meta = s_p
        t_dec_bboxes, t_dec_scores, t_enc_bboxes, t_enc_scores, t_dn_meta = t_p
        
        if s_dn_meta is not None:
            _, s_dec_bboxes = torch.split(s_dec_bboxes, s_dn_meta['dn_num_split'], dim=2)
            _, s_dec_scores = torch.split(s_dec_scores, s_dn_meta['dn_num_split'], dim=2)
            # (decoder_layer, bs, 300, 4) (decoder_layer, bs, 300, cls_num)
            s_dec_bboxes, s_dec_scores = s_dec_bboxes[-1:], s_dec_scores[-1:]
        if t_dn_meta is not None:
            _, t_dec_bboxes = torch.split(t_dec_bboxes, t_dn_meta['dn_num_split'], dim=2)
            _, t_dec_scores = torch.split(t_dec_scores, t_dn_meta['dn_num_split'], dim=2)
            t_dec_bboxes, t_dec_scores = t_dec_bboxes[-1:], t_dec_scores[-1:]
        
        t_obj_scale = t_dec_scores.sigmoid().max(-1)[0].unsqueeze(-1)
        
        lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * t_obj_scale.repeat((1, 1, 1, 4))
        # lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * self.power_transform(t_obj_scale).repeat((1, 1, 1, 4))
        lbox_iou = (1.0 - bbox_iou(s_dec_bboxes, t_dec_bboxes, xywh=True, GIoU=True)) * self.power_transform(t_obj_scale)
        lcls = nn.BCEWithLogitsLoss(reduction='none')(s_dec_scores, t_dec_scores.sigmoid()).mean(2)
        
        lbox_l1 = lbox_l1.sum() / batch['bboxes'].size(0) * 5
        lbox_iou = lbox_iou.sum() / batch['bboxes'].size(0) * 2
        lcls = lcls.sum() / (batch['bboxes'].size(0) / t_obj_scale.size(2))
        
        return lbox_l1 + lbox_iou + lcls

class RTDETRMutilDecoderLogicLoss(nn.Module):
    def __init__(self, hyp):
        super().__init__()

        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loss_ratio = torch.from_numpy(np.array([0.2, 0.5, 0.5, 1.0])).to(self.device)
        self.loss_ratio = self.loss_ratio.reshape((1, -1, 1, 1))
    
    def power_transform(self, array, power=2):
        return torch.where(array < 0.5, array ** power, array ** (1/power))
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(s_p[0].size(1), device=self.device)  # class loss
        lbox_iou = torch.zeros(s_p[0].size(1), device=self.device)  # box iou loss
        lbox_l1 = torch.zeros(s_p[0].size(1), device=self.device)  # box l1 loss
        
        s_dec_bboxes, s_dec_scores, s_enc_bboxes, s_enc_scores, s_dn_meta = s_p
        t_dec_bboxes, t_dec_scores, t_enc_bboxes, t_enc_scores, t_dn_meta = t_p
        
        if s_dn_meta is not None:
            _, s_dec_bboxes = torch.split(s_dec_bboxes, s_dn_meta['dn_num_split'], dim=2)
            _, s_dec_scores = torch.split(s_dec_scores, s_dn_meta['dn_num_split'], dim=2)
            # s_dec_bboxes, s_dec_scores = s_dec_bboxes[-1:], s_dec_scores[-1:]
        if t_dn_meta is not None:
            _, t_dec_bboxes = torch.split(t_dec_bboxes, t_dn_meta['dn_num_split'], dim=2)
            _, t_dec_scores = torch.split(t_dec_scores, t_dn_meta['dn_num_split'], dim=2)
        
        # concat encoder and decoder for teacher and student
        t_dec_bboxes, t_dec_scores = torch.cat([t_enc_bboxes.unsqueeze(0), t_dec_bboxes]), torch.cat([t_enc_scores.unsqueeze(0), t_dec_scores])
        s_dec_bboxes, s_dec_scores = torch.cat([s_enc_bboxes.unsqueeze(0), s_dec_bboxes]), torch.cat([s_enc_scores.unsqueeze(0), s_dec_scores])
        
        if t_dec_bboxes.size(0) != s_dec_bboxes.size(0):
            raise Exception(f"teacher:{t_dec_bboxes.size(0)} and student:{s_dec_bboxes.size(0)} decode layers not equal.")
        
        t_obj_scale = t_dec_scores.sigmoid().max(-1)[0].unsqueeze(-1)
        
        lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * t_obj_scale.repeat((1, 1, 1, 4))
        lbox_iou = (1.0 - bbox_iou(s_dec_bboxes, t_dec_bboxes, xywh=True, GIoU=True)) * self.power_transform(t_obj_scale)
        lcls = nn.BCEWithLogitsLoss(reduction='none')(s_dec_scores, t_dec_scores.sigmoid()).mean(2)
        
        lbox_l1 = lbox_l1 / batch['bboxes'].size(0) * 5
        lbox_iou = lbox_iou / batch['bboxes'].size(0) * 2
        lcls = lcls / (batch['bboxes'].size(0) / t_obj_scale.size(2))
        if self.loss_ratio.size(1) == lbox_l1.size(1):
            lbox_l1 = (lbox_l1 * self.loss_ratio).sum()
            lbox_iou = (lbox_iou * self.loss_ratio).sum()
            lcls = (lcls * self.loss_ratio).sum()
        else:
            lbox_l1 = lbox_l1.mean(1).sum()
            lbox_iou = lbox_iou.mean(1).sum()
            lcls = lcls.mean(1).sum()
        
        return lbox_l1 + lbox_iou + lcls


class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd'):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device) if channel != tea_channel else nn.Identity()
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if (distiller == 'mimic'):
            self.feature_loss = MimicLoss(channels_s, channels_t)

        elif (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss



class CWDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, tau=1.0, 
                 lambda_stat=0.1, lambda_channel=0.2):
        super(CWDLoss, self).__init__()
        self.tau = tau
        self.lambda_stat = lambda_stat 
        self.lambda_channel = lambda_channel  

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        total_loss = 0.0

        for s, t in zip(y_s, y_t):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            
           
            s_flat = s.view(-1, W * H) / self.tau  # [N*C, H*W]
            t_flat = t.view(-1, W * H) / self.tau
            
            softmax_t = F.softmax(t_flat, dim=1)
            log_softmax_s = F.log_softmax(s_flat, dim=1)
            log_softmax_t = F.log_softmax(t_flat, dim=1)
            
            cwd_loss = torch.sum(softmax_t * (log_softmax_t - log_softmax_s)) * (self.tau **2)
            cwd_loss = cwd_loss / (C * N)  
            
            
            s_mean = torch.mean(s, dim=[2, 3])  # [N, C]
            s_var = torch.var(s, dim=[2, 3])
            t_mean = torch.mean(t, dim=[2, 3])
            t_var = torch.var(t, dim=[2, 3])
            
            stat_loss = F.mse_loss(s_mean, t_mean) + F.mse_loss(s_var, t_var)
            
            
            s_channel = s.view(N, C, -1).mean(dim=2)  
            t_channel = t.view(N, C, -1).mean(dim=2)
            
            s_norm = F.normalize(s_channel, dim=1)  
            t_norm = F.normalize(t_channel, dim=1)
            cos_sim = torch.sum(s_norm * t_norm, dim=1).mean() 
            cos_loss = 1 - cos_sim 
            channel_loss = cos_loss  
            
          
            loss = cwd_loss + \
                   self.lambda_stat * stat_loss + \
                   self.lambda_channel * channel_loss
                   
            total_loss += loss

        return total_loss / len(y_s) 
