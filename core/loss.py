import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from utils.utils import tensor2img, img2tensor
import lpips
from pytorch_msssim import SSIM, MS_SSIM


class RenderingLoss():
    def __init__(self, cfg):
        self.mse_loss = nn.MSELoss()     
        self.l1_loss = nn.L1Loss()
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)
        
        self.cfg = cfg

    def photometric_loss(self, im1, im2):
        loss = (im1 - im2).abs()
        loss = loss.mean()
        return loss
    
    def psnr(self, im1, im2):
        mse = torch.mean((im1*255 - im2*255) ** 2).int()
        return 20 * torch.log10(255 / torch.sqrt(mse))

    def silhouette_loss(self, coarse_mask, gt_mask):
        return self.bce_loss(coarse_mask, gt_mask)

    def uv_loss(self, vis_uv, invis_uv, gt_vis_uv, html_uv):
        v_loss = self.l1_loss(vis_uv, gt_vis_uv)
        i_loss = self.l1_loss(invis_uv, html_uv * (invis_uv > 0))

        return v_loss + i_loss
    
    
    def ren_loss(self, outputs, overlap_mask=None):
        input_hand = outputs['input_hand'] 
   
        reconstructed_hand = outputs['recon_hand']
        gt_hand = input_hand
        overlap_mask = torch.ones_like(outputs['recon_hand']).to(input_hand.device) if overlap_mask is None else overlap_mask
        
        loss_dict = {}
        
        total_loss = 0 
        recon_loss = self.l1_loss(reconstructed_hand * overlap_mask, gt_hand * overlap_mask)
        sym_loss = self.l1_loss(outputs['full_tex_1d_l'], outputs['full_tex_1d_r']) 
        total_loss += recon_loss + 0.3 * sym_loss
        
        if self.cfg.MODEL.COARSE_ESTIMATION:
            alb_loss = self.photometric_loss(outputs['alb_recon_hand'], input_hand) 
            alb_con_loss = self.l1_loss(outputs['alb_recon_hand'], input_hand) + 0.15 * self.photometric_loss(outputs['rl_alb_im'], outputs['rl_alb_im2']) + 0.2 * (self.photometric_loss(outputs['coarse_alb_im'], outputs['rl_alb_im'])  + 0.2 * self.photometric_loss(outputs['coarse_alb_im'], outputs['rl_alb_im2'])) 
            coarse_recon_loss = self.l1_loss(outputs['coarse_hand'] * overlap_mask, input_hand * overlap_mask)
            invisible_mask = [1 - outputs['visible_1d_mask'][0], 1 - outputs['visible_1d_mask'][1]]
            nv_loss = self.l1_loss(outputs['full_tex_1d_l'] * invisible_mask[0], outputs['coarse_1d_l'] * invisible_mask[0]) + self.l1_loss(outputs['full_tex_1d_r'] * invisible_mask[1], outputs['coarse_1d_r'] * invisible_mask[1])
            uv_loss = self.l1_loss(outputs['full_tex_1d_l'], outputs['coarse_1d_l']) + self.l1_loss(outputs['full_tex_1d_r'], outputs['coarse_1d_r'])
            
            loss_dict['novel_view_loss'] =  nv_loss.item()
            loss_dict['coarse_recon_loss'] = coarse_recon_loss.item()
            loss_dict['coarse_alb_loss'] =  alb_loss.item()
            
            total_loss += 0.4 * alb_loss + 0.2 * alb_con_loss + 0.8 * coarse_recon_loss + 0.2 * nv_loss + 0.1 * uv_loss          
        
        # Benchmark
        reconstructed_hand_bench = img2tensor(tensor2img(reconstructed_hand), img_size=256, device=reconstructed_hand.device).unsqueeze(0) 
        gt_hand_bench = img2tensor(tensor2img(gt_hand), img_size=256, device=gt_hand.device).unsqueeze(0)
        
        recon_loss = self.l1_loss(reconstructed_hand_bench, gt_hand_bench) 
        lpips_loss = self.lpips(reconstructed_hand_bench, gt_hand_bench).mean()
        msssim_loss = self.ms_ssim(reconstructed_hand_bench, gt_hand_bench)       
        ssim_loss = self.ssim(reconstructed_hand_bench, gt_hand_bench)       
        psnr_loss = self.psnr(reconstructed_hand_bench, gt_hand_bench)

        loss_dict['recon_loss'] = recon_loss.item()
        loss_dict['psnr_loss'] = psnr_loss.item()
        loss_dict['lpips_loss'] = lpips_loss.item()
        loss_dict['msssim_loss'] = msssim_loss.item()
        loss_dict['ssim_loss'] = ssim_loss.item()


        loss_dict['total'] = total_loss

        return loss_dict
    
    def eval_loss(self, recon_hand, input_hand):
        reconstructed_hand_bench = img2tensor(tensor2img(recon_hand), img_size=512, device=recon_hand.device).unsqueeze(0)
        gt_hand_bench = img2tensor(tensor2img(input_hand), img_size=512, device=input_hand.device).unsqueeze(0)
        
        recon_loss = self.l1_loss(recon_hand, input_hand)

        lpips_loss = self.lpips(reconstructed_hand_bench, gt_hand_bench).mean()
        msssim_loss = self.ms_ssim(reconstructed_hand_bench, gt_hand_bench)       
        ssim_loss = self.ssim(reconstructed_hand_bench, gt_hand_bench)       
        psnr_loss = self.psnr(reconstructed_hand_bench, gt_hand_bench)

        loss_dict = {}
        loss_dict['recon_loss'] = recon_loss.item()
        loss_dict['lpips_loss'] = lpips_loss.item()
        loss_dict['psnr_loss'] = psnr_loss.item()
        loss_dict['msssim_loss'] = msssim_loss.item()
        loss_dict['ssim_loss'] = ssim_loss.item()

        return loss_dict
    