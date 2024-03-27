import torch
import torch.nn.functional as F
import torch.nn as nn 
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader,
    BlendParams,
)
from pytorch3d.renderer.mesh.shading import _apply_lighting
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
from pytorch3d.ops import interpolate_face_attributes
from HTML.utils.HTML import HTML

from utils.utils import *
import numpy as np
import math
import torchvision.transforms as T

from models.manolayer import ManoLayer

def phong_shading_with_shadow(meshes, fragments, lights, cameras, materials, texels, vis_maps):
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )

    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    # colors = (ambient + diffuse) * texels + specular
    vis_maps = vis_maps.unsqueeze(-2).repeat(1, 1, 1, diffuse.shape[3], 1)
    colors = (ambient + diffuse * vis_maps) * texels + specular
    
    return colors

class SoftPhongShaderShadow(nn.Module):
    def __init__(self, 
            device="cpu", cameras=None, lights=None, materials=None, blend_params=None
        ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs):
        cameras = self.cameras
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = self.lights
        materials = self.materials
        blend_params = self.blend_params
        vis_maps = kwargs['vis_maps']

        colors = phong_shading_with_shadow(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
            vis_maps=vis_maps
        )
        znear = 1.0
        zfar = 100.0
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images





class RenderModule: 
    def __init__(self, mano, tex_path=None, uv_path=None, img_size=224, tex_img_size=512, batch_size=4, device='cpu'):
        super(RenderModule, self).__init__()

        self.image_size = img_size
        self.tex_img_size = tex_img_size
        self.batch_size = batch_size
        self.device = device

        self.oneD_size = 206330
        self.mano = mano
        self.mano['left'].to(self.device)
        self.mano['right'].to(self.device)

        self.left_faces = torch.from_numpy(self.mano['left'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.right_faces = torch.from_numpy(self.mano['right'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.faces = torch.cat((self.left_faces, self.right_faces + 778), dim=1)
        
        t_left_faces = torch.from_numpy(self.mano['left'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        t_right_faces = torch.from_numpy(self.mano['right'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        t_left_faces = t_right_faces[..., [1, 0, 2]]

        self.t_faces = torch.cat((t_left_faces, t_right_faces + 778), dim=1)


        self.faces_list = [self.left_faces, self.right_faces]

        self.hand_type = None

        self.l_html = HTML(
            tex_path=tex_path, 
            uv_path=uv_path['left'], 
            device = self.device, 
            tex_image_size=tex_img_size,
            batch_size = self.batch_size
        )
        self.r_html = HTML(
            tex_path=tex_path, 
            uv_path=uv_path['right'], 
            device = self.device, 
            tex_image_size=tex_img_size,
            batch_size = self.batch_size
        )
        self.htmls = [self.l_html, self.r_html]

        self.verts_uvs = [self.l_html.verts_uvs.unsqueeze(0).repeat((self.batch_size, 1, 1)).to(self.device) , self.r_html.verts_uvs.unsqueeze(0).repeat((self.batch_size, 1, 1)).to(self.device)]
        self.faces_uvs = [self.l_html.faces_uvs.unsqueeze(0).repeat((self.batch_size, 1, 1)).to(self.device), self.r_html.faces_uvs.unsqueeze(0).repeat((self.batch_size, 1, 1)).to(self.device)]

    def reset_hand(self,):
        self.hand_type = []

    def set_hand(self, hand_types=[0, 1]):
        self.hand_type = hand_types

    def vert2mesh(self, verts, hand_type):
        return Meshes(verts=verts, faces=self.faces_list[hand_type])
    
    def change_view(self, cameras, cam_dir, trans=[0, 0, 0]):
        
        face = cam_dir[3]
        alpha = cam_dir[0] * face
        beta =  cam_dir[1] * face
        gamma = cam_dir[2] * face

        R_x = torch.tensor(
            [[-1, 0, 0], 
            [0, math.cos(alpha), -math.sin(alpha)], 
            [0, math.sin(alpha), math.cos(alpha)]]
        )
        R_y = torch.tensor(
            [[math.cos(beta), 0, math.sin(beta)], 
            [0, -1, 0], 
            [-math.sin(beta), 0, math.cos(beta)]]
        )
        R_z = torch.tensor(
            [[math.cos(gamma), -math.sin(gamma), 0], 
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, face]]
        )
        R = (R_x @ R_y @ R_z).repeat(self.batch_size, 1, 1).to(cameras.focal_length.dtype)
        T = torch.tensor(trans).repeat(self.batch_size, 1).to(cameras.focal_length.dtype)
        
        focal_length = cameras.focal_length
        principal_point = cameras.principal_point
        # import pdb; pdb.set_trace()
        cameras = OrthographicCameras(
            focal_length =focal_length,
            principal_point=principal_point,
            R=R.to(self.device),
            T=T.to(self.device), 
            in_ndc=True, 
            image_size=self.image_size,
            device=self.device
        )

        return cameras
    
    def build_camera(self, scale_left=None, scale_right=None, trans2d_left=None, trans2d_right=None, cam_dir=[0, 0, 0, 1]):
        # alpha : z axis
        # beta : x axis
        # gamma : rotation angle
        # face : direction 1 --> front, -1 --> back
        cameras = None

        scale_right = scale_right
        scale_left = scale_left
        trans2d_right = trans2d_right
        trans2d_left = trans2d_left

        scale = scale_left
        trans2d = trans2d_left

        face = cam_dir[3]
        alpha = cam_dir[0] * face
        beta =  cam_dir[1] * face
        gamma = cam_dir[2] * face

        R_x = torch.tensor(
            [[-1, 0, 0], 
            [0, math.cos(alpha), -math.sin(alpha)], 
            [0, math.sin(alpha), math.cos(alpha)]]
        )
        R_y = torch.tensor(
            [[math.cos(beta), 0, math.sin(beta)], 
            [0, -1, 0], 
            [-math.sin(beta), 0, math.cos(beta)]]
        )
        R_z = torch.tensor(
            [[math.cos(gamma), -math.sin(gamma), 0], 
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, face]]
        )
    
        R = (R_x @ R_y @ R_z).repeat(self.batch_size, 1, 1).to(scale.dtype)
        T = torch.tensor([0, 0, 10]).repeat(self.batch_size, 1).to(scale.dtype)
        cameras = OrthographicCameras(
            focal_length=2 * scale.to(self.device),
            principal_point=-trans2d.to(self.device),
            R=R.to(self.device),
            T=T.to(self.device), 
            in_ndc=True, 
            device=self.device
        )
        
        return cameras, scale_left, scale_right, trans2d_left, trans2d_right


    def build_perspective_camera(self, cameras, cam_dir=[0, 0, 0, 1], trans=[0, 0, 0]):
        fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / self.image_size
        pps = -cameras[:, :2, -1] * 2 / self.image_size + 1

        
        face = cam_dir[3]
        alpha = cam_dir[0] * face
        beta =  cam_dir[1] * face
        gamma = cam_dir[2] * face


        R_x = torch.tensor(
            [[1, 0, 0], 
            [0, math.cos(alpha), math.sin(alpha)], 
            [0, math.sin(alpha), math.cos(alpha)]]
        )
        R_y = torch.tensor(
            [[math.cos(beta), 0, math.sin(beta)], 
            [0, 1, 0], 
            [math.sin(beta), 0, math.cos(beta)]]
        )
        R_z = torch.tensor(
            [[math.cos(gamma), math.sin(gamma), 0], 
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, face]]
        )
    
        R = (R_x @ R_y @ R_z).repeat(self.batch_size, 1, 1).to(fs.dtype)
        T = torch.tensor(trans).repeat(self.batch_size, 1).to(fs.dtype)

        # db; pdb.set_trace()
        return PerspectiveCameras(focal_length=fs.to(self.device),
                                      principal_point=pps.to(self.device),
                                      in_ndc=True,
                                      R=R,
                                      T=T,
                                      device=self.device,
                                      image_size=self.image_size
                                      )

    def build_light(self, 
        light_color=((0.5, 0.5, 0.5),), 
        light_diff=((0.1, 0.1, 0.1),), 
        light_spec=((0.3, 0.3, 0.3),), 
        light_location=((0, 1, 0),)
    ):
        return PointLights(
            ambient_color=light_color, # 3
            diffuse_color=light_diff, # 3
            specular_color=light_spec,  # 3
            location=light_location, # 3
            device=self.device
        )
         
        
    def build_renderer(self, cameras, lights):
        material = Materials(device=self.device, ambient_color=[[1, 1, 1]], diffuse_color=[[1, 1, 1]], specular_color=[[0, 0, 0]])

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=0
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=self.device,
                lights=lights,
                materials = material,
                cameras = cameras,
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
            )
        )
        
        return rasterizer, renderer
    
    def trans_vert(self, v3d_left, v3d_right, scale_left, scale_right, trans2d_left, trans2d_right):
        scale_left = scale_left
        trans2d_left = trans2d_left

        scale_right = scale_right
        trans2d_right = trans2d_right

        s = scale_right / scale_left
        d = -(trans2d_left - trans2d_right) / 2 / scale_left.unsqueeze(-1)

        s = s.unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(1)

        v3d_right = s * v3d_right
        v3d_right[..., :2] = v3d_right[..., :2] + d

        return  v3d_left, v3d_right
    
    def uv2oneD(self, tex_uv_list):
        tex_1d_list = [None, None]
        for hand in self.hand_type:
            tex_1d_list[hand] = self.htmls[hand].uv21d(tex_uv_list[hand])
        return tex_1d_list

    def oneD2uv(self, tex_1d_list):
        tex_uv_list = [None, None]
        for hand in self.hand_type:
            tex_uv_list[hand] = self.htmls[hand].vec2img(tex_1d_list[hand].float().reshape(self.batch_size, -1))
        return tex_uv_list

    def uv2tex(self, tex_uv_list):
        texture_list = [None, None]
        for hand in self.hand_type:
            texture_list[hand] = TexturesUV(verts_uvs=self.verts_uvs[hand], faces_uvs=self.faces_uvs[hand], maps=tex_uv_list[hand].permute(0, 2, 3, 1))

        return texture_list
    
    def vec2tex(self, tex_vec_list, use_tex_mean=True):
        tex_uv_list = [None, None]
        tex_1d_list = [None, None]
        for hand in self.hand_type:
            tex_uv_list[hand] = self.htmls[hand].get_mano_texture(tex_vec_list[hand], use_tex_mean=use_tex_mean).clamp(0, 1)
            tex_1d_list[hand] = self.htmls[hand].uv21d(tex_uv_list[hand])
            
        texture_list = self.uv2tex(tex_uv_list)
        
        return tex_uv_list, texture_list, tex_1d_list

    def randVec(self, len, range_scale):
        rand_val_list = [None, None]
        for hand in self.hand_type:
            rand_val_list[hand] = (range_scale*(torch.rand(len) - 0.5)).unsqueeze(0).repeat(self.batch_size, 1)

        return rand_val_list

    def set_texture(self, mesh_list, texture_list=None):
        if texture_list is None:
            rand_val_list = self.randVec(101, 3)
            tex_uv_list, texture_list, _ = self.vec2tex(rand_val_list)

        for hand in self.hand_type:
            mesh_list[hand].textures = texture_list[hand]

        return mesh_list

    def render_rgb(self, mesh_list, renderer, texture_list=None):        
        imgs = [None, None, None]
        if texture_list is None:
            rand_val_list = self.randVec(101, 3)
            tex_uv_list, texture_list, _ = self.vec2tex(rand_val_list)

        for hand in self.hand_type:
            mesh_list[hand].textures = texture_list[hand]
            imgs[hand] = torch.clamp(renderer(mesh_list[hand])[..., :3].permute(0, 3, 1, 2), 0, 1)
        
        if len(self.hand_type) == 2:
            mesh = join_meshes_as_scene(mesh_list, include_textures=True)
            imgs[-1] = torch.clamp(renderer(mesh)[..., :3].permute(0, 3, 1, 2), 0, 1)
        
        return imgs
    
    def render_vertex(self, mesh_list, verts_list, renderer, texture_list):
        v_color = torch.zeros((1, 778 * 2, 3))
        v_color[0, :778] = texture_list[0]
        v_color[0, 778:] = texture_list[1]
        v_color.repeat(self.batch_size, 1, 1).int()
     
        textures = TexturesVertex(verts_features=v_color.to(self.device))
        v3d = torch.cat((verts_list[0], verts_list[1]), dim=1)
        mesh = Meshes(verts=v3d.to(self.device), faces=self.faces.to(self.device), textures=textures)
        gamma =  renderer(mesh)[..., :3].permute(0, 3, 1, 2) / 255.
        
        return torch.clamp(gamma, 0, 1)
    
    
    def render_sil(self, mesh_list, cameras, rasterizer):
        depth_list = self.get_depth_map(mesh_list, cameras, rasterizer)
        l_dp = depth_list[0]
        r_dp = depth_list[1]
        both_dp = depth_list[2]
        
        l_sil = (both_dp != r_dp).long().repeat(1, 3, 1, 1)
        r_sil = (both_dp != l_dp).long().repeat(1, 3, 1, 1)
        both_sil = (both_dp != 0).long().repeat(1, 3, 1, 1)

        return [l_sil, r_sil, both_sil]

    def unwrap_img2uv(self, img, mesh, rasterizer, cameras, mask=None, hand_id=0):

        Resizer = T.Resize(self.tex_img_size)

        fragments = rasterizer(mesh, cameras=cameras)
        # verts_uvs_list : [907, 2] [[0.398, 0.142], [], ..., []]
        # faces_uvs_list : [1538, 3] [[0, 1, 906], [], ..., []]]
        faces_verts_uvs = torch.cat([
            i[j] for i, j in zip(mesh.textures.verts_uvs_list(), mesh.textures.faces_uvs_list())
        ])

        texture_maps = mesh.textures.maps_padded()
        N, H_in, W_in, C = texture_maps.shape
        N, H_out, W_out, K = fragments.pix_to_face.shape
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
        inv_pixel_uvs = torch.zeros((N * K, self.image_size, self.image_size, 2)).to(self.device)

        x_indices = (pixel_uvs[:, :, :, 0] * (self.image_size - 1)).long()
        y_indices = (pixel_uvs[:, :, :, 1] * (self.image_size - 1)).long()
        idxs = np.array(np.where(pixel_uvs.cpu().detach().numpy()))[:3]

        kk = np.zeros((N, self.image_size, self.image_size, 2))
        kk[idxs[0], idxs[1], idxs[2]] = np.array([idxs[2], idxs[1]]).transpose([1, 0])  

        for b in range(N):
            inv_pixel_uvs[b, y_indices[b], x_indices[b]] = torch.tensor(kk[b]).float().to(self.device) / (self.image_size- 1)
        
        inv_pixel_uvs =  inv_pixel_uvs * 2.0 - 1.0

        texels = F.grid_sample(
            img,
            inv_pixel_uvs,
            mode='bilinear',
            align_corners=True,
            padding_mode='border',
        )
        texels = Resizer(torch.flip(texels, [2]))
        tex_1d = self.htmls[hand_id].uv21d(texels).reshape((self.batch_size, -1, 3))
    
        if mask is not None:
            masked_img = img * mask + torch.ones_like(img) *  (1 - mask.clone())
            masked_texels = F.grid_sample(
                masked_img,
                inv_pixel_uvs,
                mode='bilinear',
                align_corners=True,
                padding_mode='border',
            )
            masked_texels = Resizer(torch.flip(masked_texels, [2]))
            same = (texels == masked_texels)
            refined_texels = texels* same
            tex_1d = self.htmls[hand_id].uv21d(refined_texels)
            return Resizer(texels * same), tex_1d
        
        
        return Resizer(texels), tex_1d
    
    def get_depth_map(self, mesh_list, cameras, rasterizer):
        dp_list = [
            torch.zeros(self.batch_size, 1, self.image_size, self.image_size).to(self.device) - 1, 
            torch.zeros(self.batch_size, 1, self.image_size, self.image_size).to(self.device) - 1
        ]

        # import pdb; pdb.set_trace()
        for hand in self.hand_type:
            hand_mesh = mesh_list[hand]
            fragments = rasterizer(hand_mesh, cameras=cameras)
            dp_list[hand] = fragments.zbuf.permute(0, 3, 1, 2)         
           
        left_dp = dp_list[0]
        right_dp = dp_list[1]
        min_dp = min(left_dp.unique()[1], right_dp.unique()[1])

        left_dp[left_dp > -1] = left_dp[left_dp > -1] - min_dp
        right_dp[right_dp > -1] = right_dp[right_dp > -1] - min_dp

        max_dp = max(left_dp.max(), right_dp.max())

        left_dp[left_dp > -1] = 1 - 0.7 * left_dp[left_dp > -1] / max_dp
        right_dp[right_dp > -1] = 1 - 0.7 * right_dp[right_dp > -1] / max_dp

        left_dp[left_dp == -1]   = 0
        right_dp[right_dp == -1] = 0

        both_dp = torch.maximum(left_dp, right_dp)
        
        return [left_dp, right_dp, both_dp]
        
    def get_normal_map(self, mesh_list, cameras, rasterizer):
        # blend_params = BlendParams(background_color=(1.0,1.0,1.0))
        nm_list = [
            torch.zeros(self.batch_size, 3, self.image_size, self.image_size).to(self.device), 
            torch.zeros(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        ]

        for hand in self.hand_type:
            hand_mesh = mesh_list[hand]
            fragments = rasterizer(hand_mesh, cameras=cameras)
            mask = fragments.zbuf.permute(0, 3, 1, 2)
            mask = (mask != -1).long().repeat(1, 3, 1, 1)

            faces = hand_mesh.faces_packed()  # (F, 3)
            vertex_normals = hand_mesh.verts_normals_packed()  # (V, 3)
            faces_normals = vertex_normals[faces] # (F, 3, 3)
            
            pixel_normals = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, faces_normals
            )

            pixel_normals[:, :, :, :, 1] = pixel_normals[:, :, :, :, 1] * -1.
            pixel_normals[:, :, :, :, 2] = pixel_normals[:, :, :, :, 2] * -1.
            pixel_normals = (pixel_normals + 1.0) / 2.0
            
            pixel_normals = pixel_normals.squeeze(-2).permute(0, 3, 1, 2)
            # import pdb; pdb.set_trace()
            pixel_normals[mask == 0] = 0
            nm_list[hand] = pixel_normals
            
        sil_list = self.render_sil(mesh_list, cameras, rasterizer)
        
        both_nm = sil_list[0] * nm_list[0] + sil_list[1] * nm_list[1]  
        nm_list.append(both_nm)

        return nm_list

    def build_shadow_renderer(self, cameras, lights):
        material = Materials(device=self.device, ambient_color=[[1, 1, 1]], diffuse_color=[[1, 1, 1]], specular_color=[[0, 0, 0]])

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=0
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        shader=SoftPhongShaderShadow(
            device=self.device,
            lights=lights,
            materials = material,
            cameras = cameras,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
        
        return rasterizer, shader

    def render_shadow(self, mesh_list, fragments, shader, texture_list, **kwargs):
        # meshes = mesh_list[0]
        imgs = [None, None, None]
        for hand in self.hand_type:
            mesh_list[hand].textures = texture_list[hand]

        if len(self.hand_type) == 2:
            mesh = join_meshes_as_scene(mesh_list, include_textures=True)
            imgs[-1] = shader(fragments, mesh, **kwargs)[..., :3].permute(0, 3, 1, 2)

        return imgs
    


def load_interHandRender(cfg, device):   
    mano_path = get_mano_path(cfg)
    mano = {'right': ManoLayer(mano_path['right'], center_idx=None), 'left': ManoLayer(mano_path['left'], center_idx=None)}
    
    hand_renderer = RenderModule(
        mano,
        tex_path=get_tex_path(cfg),
        uv_path=get_uvs_path(cfg),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        img_size=cfg.IMG_SIZE,
        tex_img_size=cfg.UV_SIZE,
        device=device
    )

    return hand_renderer