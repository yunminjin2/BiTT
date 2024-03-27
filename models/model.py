import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.encoder import load_encoder
from models.decoder import load_decoder
from models.network import *

from core.loss import *

from utils.config import load_cfg
from utils.utils import *

from models.InterHandRender import *


def freeze_model(model):
    for (name, params) in model.named_parameters():
        params.requires_grad = False

class IntagNet(nn.Module):
    def __init__(self, encoder, mid_model, decoder):
        super(IntagNet, self).__init__()
        self.encoder = encoder
        self.mid_model = mid_model
        self.decoder = decoder

    def forward(self, img):
        resizer = transforms.Resize(256)
        img = resizer(img)
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(img)
        global_feature, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)
        result, paramsDict, handDictList, otherInfo = self.decoder(global_feature, fmaps)

        if hms is not None:
            otherInfo['hms'] = hms
        if mask is not None:
            otherInfo['mask'] = mask
        if dp is not None:
            otherInfo['dense'] = dp

        return result, paramsDict, handDictList, otherInfo

    
class CoarseModel(nn.Module):
    def __init__(self, cfg, hand_render, device='cpu'):
        super(CoarseModel, self).__init__()

        self.image_size = cfg.IMG_SIZE
        self.tex_image_size = cfg.UV_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.device = device

        self.img_processor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])     
        self.hand_render = hand_render

        self.HTMLNet = Encoder(cin=3, cout=202, in_size=256)
        self.resizer = T.Resize(256)
        
    def forward(self, input_hand):
        input_hand = self.resizer(input_hand)

        ## Run HTML Model
        B = input_hand.size(0)
        html_vec = self.HTMLNet(input_hand) * 2
        html_vec_l, html_vec_r = html_vec[:, :101], html_vec[:, 101:]
        html_uv_list, html_tex_list, html_tex_1d_list = self.hand_render.vec2tex([html_vec_l, html_vec_r])
        
        return html_uv_list, html_tex_list, html_tex_1d_list


class BiTT(nn.Module):
    def __init__(self, cfg, hand_type = 'BOTH', device='cpu'):
        super(BiTT, self).__init__()
        self.device = device
        
        self.hand_type = hand_type
        self.hand_idx = -1
        if self.hand_type == 'LEFT': self.hand_idx = 0
        elif self.hand_type == 'RIGHT': self.hand_idx = 1
        
        self.model_type = cfg.MODEL_TYPE
        self.coarse_estimation = cfg.MODEL.COARSE_ESTIMATION
        self.use_gt_shape = cfg.MODEL.USE_GT_SHAPE        
        
        self.IntagNet = None 
        if not self.use_gt_shape:
            self.IntagNet = load_intag_model(cfg)
            freeze_model(self.IntagNet)
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.image_size = cfg.IMG_SIZE
        self.tex_image_size = cfg.UV_SIZE
        
        self.basic_rendering_is_done = False
        self.do_refine_mask = cfg.REFINE_MASK
        self.did_refine_mask = False
        
        ## Renderer Init
        self.hand_render = load_interHandRender(cfg, self.device)
        self.vertex_rendering = False
        
        ## Network Init        
        self.coarse_model = load_coarse_model(cfg, hand_type=hand_type, hand_idx=self.hand_idx, hand_render=self.hand_render, device=self.device)
        self.AlbedoNet = UnetGenerator(input_nc=3, output_nc=3, num_downs=6, f_act=nn.Sigmoid)
        self.LightNet = Encoder(cin=3, cout=12, nf=32, in_size=self.image_size, activation=nn.Tanh)  
        self.fine_model = BiTextureNet(input_nc=3, output_nc=3,mid_channel=cfg.MODEL.ENCODING_CHANNEL, batch_size=self.batch_size, f_act=nn.Sigmoid, device=self.device)
            

    def get_gt_shape(self, hand_dict):
        gt_params = {}
        # import pdb; pdb.set_trace()
        if type(hand_dict['left']['camera']) == dict:
            gt_params['cameras'] = hand_dict['left']['camera']
        else:
            gt_params['cameras'] = hand_dict['left']['camera'].float().to(self.device)
        gt_params['v3d_left'] = hand_dict['left']['verts3d'].to(self.device)
        gt_params['v3d_right'] = hand_dict['right']['verts3d'].to(self.device)
                
        return gt_params

    def run_intag_model(self, input_im):
        def process_img(img):
            img_processor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            imgTensor = img_processor(img)
            return imgTensor
        
        input_im = process_img(input_im)
        result, paramsDict, handDictList, otherInfo = self.IntagNet(input_im)

        params = {}
        params['scale_left'] = paramsDict['scale']['left'].to(self.device)
        params['trans2d_left'] = paramsDict['trans2d']['left'].to(self.device)
        params['scale_right'] = paramsDict['scale']['right'].to(self.device)
        params['trans2d_right'] = paramsDict['trans2d']['right'].to(self.device)
        params['v3d_left'] = result['verts3d']['left'].to(self.device)
        params['v3d_right'] = result['verts3d']['right'].to(self.device)
        params['otherInfo'] = otherInfo
        
        return params
    
    def define_mask(self, mesh_list, cameras, rasterizer):
        mask = self.hand_render.render_sil(mesh_list, cameras, rasterizer)[self.hand_idx]
        return mask

    def set_scene(self, input_im=None, hand_dict=None):
        if self.use_gt_shape and hand_dict is not None:
            gt_params = self.get_gt_shape(hand_dict)
            l_verts = gt_params['v3d_left']
            r_verts = gt_params['v3d_right']

            cameras = self.hand_render.build_perspective_camera(gt_params['cameras'])
                
            self.gt_params = gt_params
        else:
            input_im = input_im if input_im is not None else self.input_im
            self.intag_params = self.run_intag_model(input_im)
        
            ## Consturct Scene
            cameras, scale_left, scale_right, trans2d_left, trans2d_right = self.hand_render.build_camera(
                scale_left=self.intag_params['scale_left'], 
                scale_right=self.intag_params['scale_right'], 
                trans2d_left=self.intag_params['trans2d_left'], 
                trans2d_right=self.intag_params['trans2d_right'],
            )
            l_verts, r_verts = self.hand_render.trans_vert(
                self.intag_params['v3d_left'],
                self.intag_params['v3d_right'],
                scale_left, scale_right, trans2d_left, trans2d_right
            )
        
        l_mesh, r_mesh = self.hand_render.vert2mesh(l_verts, 0), self.hand_render.vert2mesh(r_verts, 1)
        
        mesh_list = [l_mesh, r_mesh]
        verts_list = [l_verts, r_verts]
        
        self.hand_render.set_hand([0, 1])
        if self.hand_type == 'LEFT':
            self.hand_render.set_hand([0])
        elif self.hand_type == 'RIGHT': 
            self.hand_render.set_hand([1])
               
        return cameras, mesh_list, verts_list
    
    def foundation_work(self, input_im, input_hand, hand_dict, light=None):
        self.hand_render.reset_hand()
        self.input_im = input_im
        self.input_hand = input_hand
        self.hand_dict = hand_dict
        
        self.cameras, self.mesh_list, self.vert_list = self.set_scene(hand_dict=hand_dict)
        
        self.pred_light = self.run_light_model(input_im, light)
        
        self.rasterizer, self.renderer = self.hand_render.build_renderer(self.cameras, self.pred_light['lights'])    

        self.mesh_mask = self.define_mask(self.mesh_list, self.cameras, self.rasterizer)
        self.mask = self.mesh_mask
        
        if self.do_refine_mask:
            self.refined_mask = mask_refinement(input_im ,self.mask, device=self.device)
            self.do_refine_mask = False
            self.did_refine_mask = True
            self.mask = self.refined_mask
        self.input_hand = self.input_im * self.mask

    
    def run_light_model(self, input_im, light=None):
        out_light = {}
        light_notanh = self.LightNet(input_im)
        out_light['recon_light'] = recon_light = torch.tanh(light_notanh)
        if light is not None:
            self.recon_light = light
        out_light['light_color'] = recon_light[:, :3] * 0.4 + 0.6
        out_light['light_diff'] = recon_light[:, 3:6] * 0.4 + 0.6
        out_light['light_spec'] = recon_light[:, 6:9] * 0.4 + 0.6
        out_light['light_location'] =recon_light[:, 9:12]
        out_light['lights'] = self.hand_render.build_light(
            light_color=out_light['light_color'],
            light_diff=out_light['light_diff'],
            light_spec=out_light['light_spec'] ,
            light_location=out_light['light_location']
        )
        if light is not None:
            out_light['recon_light'] = light
            out_light['light_color'] = light[:, :3] * 0.8 + 0.2
            out_light['light_diff'] = light[:, 3:6] * 0.8 + 0.2
            out_light['light_spec'] = light[:, 6:9] * 0.8 + 0.2
            out_light['light_location'] =light[:, 9:12]
            out_light['lights'] = self.hand_render.build_light(
            light_color=out_light['light_color'],
            light_diff=out_light['light_diff'],
            light_spec=out_light['light_spec'] ,
            light_location=out_light['light_location']
        )

        return out_light
    
    def run_albedo_model(self, input_im, mesh_list, rasterizer, cameras):
        albedo_im = self.AlbedoNet(input_im)
        alb_uv_list = [None, None]
        alb_1d_list = [None, None]
        vis_uv_list = [None, None]
        vis_1d_list = [None, None]
        visible_uv_mask = [None, None]
        
        for hand in self.hand_render.hand_type:
            alb_uv_list[hand], alb_1d_list[hand] = self.hand_render.unwrap_img2uv(albedo_im, mesh_list[hand], rasterizer, cameras)
            vis_uv_list[hand], vis_1d_list[hand] = self.hand_render.unwrap_img2uv(input_im, mesh_list[hand], rasterizer, cameras)
            visible_uv_mask[hand] = (torch.sum(vis_uv_list[hand], dim=1) > 0).int().float().unsqueeze(1).repeat(1, 3, 1, 1)
        
        visible_mask = self.hand_render.uv2oneD(visible_uv_mask)
        alb_tex_list = self.hand_render.uv2tex(alb_uv_list)
        
        return alb_tex_list, alb_uv_list, alb_1d_list, albedo_im, visible_mask, visible_uv_mask, vis_uv_list, vis_1d_list

    def run_coarse_model(self, input_hand):
        coarse_datas = self.coarse_model(input_hand=input_hand)
        return coarse_datas
    
    def coarse_forward(self, input_im, input_hand, hand_dict=None, light=None):
        self.foundation_work(input_im, input_hand, hand_dict, light)
        
        self.mesh_list = self.hand_render.set_texture(self.mesh_list)
        
        #####  Albedo Estimation  #####
        self.alb_tex_list, self.alb_uv_list, self.alb_1d_list, self.alb_im, self.vis_mask_1d_list, self.vis_mask_uv_list, self.vis_uv_list ,self.vis_1d_list = self.run_albedo_model(self.input_hand, self.mesh_list, self.rasterizer, self.cameras)

        #####  Coarse Stage  #####
        self.coarse_uv_list, self.coarse_tex_list, self.coarse_1d_list = self.run_coarse_model(self.input_hand)
        
        self.mesh_list = self.hand_render.set_texture(self.mesh_list, texture_list=self.coarse_tex_list)

        ## Fully relight hand
        self.recon_new_hand = self.relight(
            tex_list=self.coarse_tex_list,
            camera=self.cameras,
            mesh_list=self.mesh_list,
            light_color=self.pred_light['light_color'],
            light_location=((0, -1, 0),),
        )['recon_hand']
        self.rl_alb_im = self.AlbedoNet(self.recon_new_hand)

        self.recon_new_hand2 = self.relight(
            tex_list=self.coarse_tex_list,
            camera=self.cameras,
            mesh_list=self.mesh_list,
            light_color=((0.5, 0.5, 0.5),),
            light_location=((0, 1, 0),),
        )['recon_hand']
        self.rl_alb_im2 = self.AlbedoNet(self.recon_new_hand2)


        self.coarse_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.coarse_tex_list)
        self.html_recon_hand = self.coarse_recon_list[self.hand_idx]  

        self.alb_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.alb_tex_list)
        self.vis_mask_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.hand_render.uv2tex(self.vis_mask_uv_list))[self.hand_idx]
        self.vis_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.hand_render.uv2tex(self.vis_uv_list))[self.hand_idx]
        self.alb_recon_hand = self.alb_recon_list[self.hand_idx]  
        
        self.coarse_alb_im = self.AlbedoNet(self.html_recon_hand)

        return {
            'html_recon': self.html_recon_hand,
            'alb_recon': self.alb_recon_hand,
            'html_uv': self.coarse_uv_list,
            'alb_uv': self.alb_uv_list,
            'html_tex': self.coarse_tex_list,
            'alb_im': self.alb_im,
            'rl_alb_im':  self.rl_alb_im,
            'rl_alb_im2': self.rl_alb_im2,
        }

    def run_fine_model(self, alb_1d_list=None, coarse_1d_list=None):
        if alb_1d_list == None:
            alb_1d_list = self.alb_1d_list
        if self.coarse_estimation:
            full_tex_1d_list = self.fine_model(self.alb_1d_list, self.coarse_1d_list, self.vis_mask_1d_list)
        else:
            if coarse_1d_list == None:
                coarse_1d_list = self.coarse_1d_list
            full_tex_1d_list = self.fine_model(alb_1d_list)
        full_uv_list = self.hand_render.oneD2uv(full_tex_1d_list)
        full_tex_list = self.hand_render.uv2tex(full_uv_list)
        
        return full_uv_list, full_tex_list, full_tex_1d_list
    

    def forward(self, input_im, input_hand=None, hand_dict=None, light=None):
        if self.coarse_estimation:
            self.coarse_forward(input_im, input_hand, hand_dict)
        else:
            self.foundation_work(input_im, input_hand, hand_dict, light)
            
            self.mesh_list = self.hand_render.set_texture(self.mesh_list)
            self.alb_tex_list, self.alb_uv_list, self.alb_1d_list, self.alb_im, self.vis_mask_1d_list, self.vis_mask_uv_list, self.vis_uv_list ,self.vis_1d_list = self.run_albedo_model(self.input_hand, self.mesh_list, self.rasterizer, self.cameras)
            
        if not self.basic_rendering_is_done:
            v_color = torch.zeros((778 * 2, 3))
            v_color[:778, 0] = 102
            v_color[:778, 1] = 102
            v_color[:778, 2] = 204
            v_color[778:, 0] = 254
            v_color[778:, 1] = 102
            v_color[778:, 2] = 102
            
            self.mesh_vis = self.hand_render.render_vertex(self.mesh_list, self.vert_list, self.renderer, texture_list=[v_color[:778], v_color[778:]])
            
            self.basic_rendering_is_done = True
        ###########################################
        
        ##### Fine Stage #####
        self.full_uv_list, self.full_tex_list, self.full_1d_list = self.run_fine_model()
            
        ###### Render Reconstruct Hand ######
        ## 1. Reconstructed Albedo Hand
        self.alb_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.alb_tex_list)
        self.alb_recon_hand = self.alb_recon_list[self.hand_idx]
        if self.coarse_estimation:
            self.coarse_recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.coarse_tex_list)
            self.coarse_recon_hand = self.coarse_recon_list[self.hand_idx]    

        ## 2. Reconstructed Full Texture Hand
        self.recon_list = self.hand_render.render_rgb(self.mesh_list, self.renderer, texture_list=self.full_tex_list)
        self.recon_hand = self.recon_list[self.hand_idx]
      
        ## 3. Relight hand 1
        self.recon_hand_rl = self.relight(self.full_tex_list, self.cameras, self.mesh_list)['recon_hand']
        self.rl_recon_alb_im = self.AlbedoNet((self.recon_hand_rl * 255).int() / 255.0)
        if self.coarse_estimation:
            self.coarse_hand_rl = self.relight(self.coarse_tex_list,self.cameras, self.mesh_list)['recon_hand']
        ## 4. Relight hand 2
        self.recon_hand_rl2 = self.relight(
            tex_list=self.full_tex_list,
            camera=self.cameras,
            mesh_list=self.mesh_list,
            light_color=(self.pred_light['light_color']),
            light_location=((0, 1, 0),),
        )['recon_hand']
        self.rl_recon_alb_im2 = self.AlbedoNet((self.recon_hand_rl2 * 255).int() / 255.0)
        if self.coarse_estimation:
            self.coarse_hand_rl = self.relight(self.coarse_tex_list,self.cameras, self.mesh_list)['recon_hand']

        # self.recon_hand_nv = self.novel_view(self.full_tex_list)
        # self.coarse_hand_nv = self.novel_view(self.coarse_tex_list)
        
        return self.wrap_up()
    
    def wrap_up(self,):
        data_dict = {}

        data_dict['input_im'] = self.input_im
        data_dict['input_hand'] = self.input_hand
        data_dict['mesh_visual'] = self.mesh_vis
        
        data_dict['mask'] = self.mask
        data_dict['refine_mask'] = self.refined_mask if self.did_refine_mask else None
        
        data_dict['visible_1d_mask'] = self.vis_mask_1d_list
        data_dict['visible_uv_mask'] = self.vis_mask_uv_list
        data_dict['visible_uv'] = self.vis_uv_list
        data_dict['recon_tex'] = self.full_tex_list

        if self.coarse_estimation:
            data_dict['coarse_tex'] = self.coarse_tex_list
            data_dict['coarse_hand'] = self.coarse_recon_hand
            data_dict['coarse_alb_im'] = self.coarse_alb_im
            data_dict['rl_alb_im'] = self.rl_alb_im * self.mask
            data_dict['rl_alb_im2'] = self.rl_alb_im2 * self.mask
        if not self.vertex_rendering:
            data_dict['albedo_im'] = self.alb_im * self.mask
            data_dict['alb_uv'] = self.alb_uv_list
            data_dict['alb_recon_hand'] = self.alb_recon_hand
        
        data_dict['recon_hand'] = self.recon_hand
        data_dict['recon_hand_rl'] = self.recon_hand_rl
        data_dict['recon_hand_rl2'] = self.recon_hand_rl2
        data_dict['rl_recon_alb_im'] = self.rl_recon_alb_im * self.mask
        data_dict['rl_recon_alb_im2'] = self.rl_recon_alb_im2 * self.mask
        
        if 0 in self.hand_render.hand_type:
            if self.coarse_estimation:
                data_dict['coarse_1d_l'] = process_none(self, self.coarse_1d_list, idx=0)
                data_dict['coarse_tex_uv_l'] = process_none(self, self.coarse_uv_list, idx=0)

        
            if not self.vertex_rendering:
                data_dict['full_tex_1d_l'] = process_none(self, self.full_1d_list, idx=0)
                data_dict['full_tex_uv_l'] = process_none(self, self.full_uv_list, idx=0)
                data_dict['alb_uv_l'] = process_none(self, self.alb_uv_list, idx=0)

        if 1 in self.hand_render.hand_type:
            if self.coarse_estimation:
                data_dict['coarse_1d_r'] = process_none(self, self.coarse_1d_list, idx=1)
                data_dict['coarse_tex_uv_r'] = process_none(self, self.coarse_uv_list, idx=1)
            if not self.vertex_rendering:
                data_dict['full_tex_1d_r'] = process_none(self, self.full_1d_list, idx=1)
                data_dict['full_tex_uv_r'] = process_none(self, self.full_uv_list, idx=1)
                data_dict['alb_uv_r'] = process_none(self, self.alb_uv_list, idx=1)
                
        return data_dict

    '''
    Given input image, texture, hand shape, render texture on hand shape.
    '''
    def render_forward(self, input_im, tex_list, hand_dict=None, light=None):
        self.hand_render.reset_hand()
        cameras, mesh_list, vert_list = self.set_scene(input_im, hand_dict)

        pred_light = light
        if light is None:
            pred_light = self.pred_light
        
        rasterizer, renderer = self.hand_render.build_renderer(cameras, pred_light['lights'])    
        mask = self.define_mask(mesh_list, cameras, rasterizer)
        input_hand = input_im * mask
        if self.vertex_rendering:
            recon_list = self.hand_render.render_vertex(mesh_list, vert_list, renderer, texture_list=tex_list)
        else:
            recon_list = self.hand_render.render_rgb(mesh_list, renderer, texture_list=tex_list)
        recon_hand = recon_list[self.hand_idx]  

        return input_hand, recon_hand

    def relight(
            self,
            tex_list,
            camera,
            mesh_list,
            light_color=((0.5, 0.5, 0.5),),
            light_spec=((0.2, 0.2, 0.2),),
            light_diff=((0.3, 0.3, 0.3),),
            light_location=((0, 0, -1),),
            camera_direction=[0, 0, 0, 1],
        ):
        result = {}

        lights = self.hand_render.build_light(
            light_color=light_color,
            light_spec=light_spec,
            light_diff=light_diff,
            light_location=light_location
        )
        _, renderer = self.hand_render.build_renderer(self.cameras, lights)   

        if self.vertex_rendering:
            recon_list = self.hand_render.render_vertex(mesh_list, self.vert_list, renderer, texture_list=tex_list)
        else: 
            recon_list = self.hand_render.render_rgb(mesh_list, renderer, texture_list=tex_list)
        result['recon_list'] = recon_list
        result['recon_hand'] = result['recon_list'][self.hand_idx]
        
        return result
    
    def novel_view(self, tex_list, cameras=None, camera_direction=[0.9, 0, 0, 1], trans=[0, 0, 0]):
        result = {}
        if cameras is None:
            cameras = self.hand_render.change_view(self.cameras, cam_dir=camera_direction, trans=trans)
        
        rasterizer, renderer = self.hand_render.build_renderer(cameras, self.pred_light['lights'])   
        if self.vertex_rendering:
            recon_list = self.hand_render.render_vertex(self.mesh_list, self.vert_list, renderer, texture_list=self.full_tex_list)
        else: 
            recon_list = self.hand_render.render_rgb(self.mesh_list, renderer, texture_list=tex_list)
        
        result['recon_list'] = recon_list
        result['recon_hand'] =  recon_list[self.hand_idx]

        return result

    def batch_novel_view(self, cameras):
        # first b is recon_im_new
        recon_batch = {}
        coarse_batch = {}
        recon_batch['left'] = self.recon_hand_nv['recon_list'][0]
        recon_batch['right'] = self.recon_hand_nv['recon_list'][1]
        coarse_batch['left'] = self.coarse_hand_nv['recon_list'][0] if self.coarse_estimation else None
        coarse_batch['right'] = self.coarse_hand_nv['recon_list'][1] if self.coarse_estimation else None

        for each_view in cameras:           
            nv = self.novel_view(self.full_tex_list, camera_direction=each_view, trans=[0, 0, 0 ])
            recon_batch['left'] = torch.cat([recon_batch['left'], nv['recon_list'][0]], dim=0)
            recon_batch['right'] = torch.cat([recon_batch['right'], nv['recon_list'][1]], dim=0)

            if self.coarse_estimation:
                gn = self.novel_view(self.coarse_tex_list, camera_direction=each_view)
                coarse_batch['left'] = torch.cat([coarse_batch['left'], gn['recon_list'][0]], dim=0)
                coarse_batch['right'] = torch.cat([coarse_batch['right'], gn['recon_list'][1]], dim=0)
   
    
        return recon_batch, coarse_batch


def process_none(_self, some_list, idx, dim=3):
        return some_list[idx] if some_list[idx] is not None else torch.zeros((_self.batch_size, dim, _self.image_size, _self.image_size))

def pad_background(hand, mask, background):
    return hand * mask + background * (1 - mask)

def load_model(model, pretrain_dict, part_lists=None):
    model_dict = model.state_dict()
    if part_lists is None: 
        model.load_state_dict(pretrain_dict)
    else:
        for each_p in part_lists:
            targets = {}
            for k, v in pretrain_dict.items():
                if each_p in k:
                    targets['k'] = v

        model.load_state_dict(targets)
        
    return model

def load_coarse_model(cfg, device, hand_render, hand_type='BOTH', hand_idx=2):
    return CoarseModel(cfg, hand_render=hand_render, device=device).to(device)

def load_intag_model(cfg):
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    encoder, mid_model = load_encoder(cfg)
    decoder = load_decoder(cfg, mid_model.get_info())
    model = IntagNet(encoder, mid_model, decoder)
    
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, str(cfg.INTAGHAND_PATH.MODEL_PRETRAIN_PATH))
    
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        print('load intag params from {}'.format(path))
        try:
            model.load_state_dict(state)
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            model.load_state_dict(state2)

    return model

def load_myModel(cfg, device, end_to_end=False):
    model = BiTT(cfg, end_to_end=end_to_end, device=device).to(device)
    return model