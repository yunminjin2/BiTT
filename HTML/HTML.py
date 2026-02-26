'''
Copyright 2020
Neng Qian, Jiayi Wang, Franziska Mueller, Florian Bernard,
Vladislav Golyanik, Christian Theobalt, and the Max Planck Institute.
All rights reserved.

This software is provided for research purposes only.
By using this software you agree to the terms of the HTML Model license.

More information about the HTML is available at https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/

Acknowledgements:
The code file is based on the release code of ICCV HAND2019 challenge with adaptations.
Check https://sites.google.com/view/hands2019/challenge
Therefore, we would like to kindly thank Dr. Anil Armagan.


Please Note:
============
This is a demo version of the script for driving the HTML, hand texture model with python.
We would be happy to receive comments, help and suggestions on improving this code
and in making it available on more platforms.


System Requirements:
====================
Operating system: OSX, Linux, Windows

Python Dependencies:
- Numpy
- OpenCV
- pytorch
- pytorch3D == 0.1


About the Script:
=================
This script demonstrates how to generate and render 3D textured hand mesh
with our HTML and MANO model to help users get started with using the HTML
model. The code shows how to:
  - Apply the HTML hand texture model to the MANO hand model
  - Edit pose, shape, texture parameters of the model to create a new 3D hand
  - Render the 3D hand mesh by a differentialable renderer Pytorch3D
  - Save the resulting texture as a 2D rgb image mesh in .png format
  - Save the rendered hand image as a 2D rgb image mesh in .png format
  - The texture can be visualized by opening the ./vis.obj in MeshLab

Note:
  - This script requires the ./MANO_RIGHT.pkl .
  Download the MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/
  - The UV coordinators (./TextureBasis/uvs.pkl) is only for the MANO right hand mesh
  For the left hand, the col of faces_uvs may need to be swapped
'''

import numpy as np
import pickle

import torch
import torchvision.transforms as T
from .MANO_SMPL import MANO_SMPL



class HTML():
    def __init__(self, tex_path, uv_path, tex_image_size, batch_size, device='cpu'):
        super(HTML, self).__init__()
        self.device = device
        self.batch_size = batch_size
        with open(tex_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(uv_path,'rb') as f:
            self.uvs = pickle.load(f)

        self.faces_uvs = torch.tensor(self.uvs['faces_uvs'], dtype = torch.long).to(self.device)
        self.verts_uvs = torch.tensor(self.uvs['verts_uvs'],dtype = torch.float).to(self.device)

        self.tex_mean = torch.tensor(np.array(self.model['mean']), dtype=torch.float).to(self.device)
        self.tex_basis = torch.tensor(self.model['basis'], dtype=torch.float).to(self.device)
        self.vec2texImg_index = torch.tensor(self.model['index_map'], dtype=torch.long).to(self.device)

        self.tex_num_pca_comp = 101
        self.tex_path = tex_path
        
        self.tex_image_size = tex_image_size      

        self.resizer = T.Resize((self.tex_image_size, self.tex_image_size))

    def vec2img(self, vec):
        img1d = torch.zeros([self.batch_size, 1024 * 1024 * 3]).to(self.device)
        for e in range(self.batch_size):
            img1d[e, self.vec2texImg_index] = vec[e]

        img1d = img1d.reshape((self.batch_size, 3, 1024, 1024)).permute((0, 1, 3, 2))
        return self.resizer(img1d)

    def get_mano_texture(self, gamma, use_tex_mean=True):
        # tex_basis [B, 618990, 101]
        # gamma [B, 101, 1]
        # offsets = [B, 618990, 1]
        # tex_mean = [B, 618990, 1]
        tex_basis = self.tex_basis.repeat(self.batch_size, 1, 1)
        tex_mean = self.tex_mean[..., None].repeat(self.batch_size, 1, 1)
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=torch.float)
        
        gamma = gamma[..., None].to(self.device)
        offsets = torch.bmm(tex_basis, gamma)

        tex_code = offsets
        if use_tex_mean:
            tex_code += tex_mean # [B, 618990, 1]
        else:
            tex_code *= 255
        new_tex_img = self.vec2img(tex_code.squeeze(-1)) / 255
        
        return new_tex_img
    
    def uv21d(self, uv):
        uv1d = uv.permute(0, 1, 3, 2).reshape((self.batch_size, uv.shape[1] * uv.shape[2] * uv.shape[3]))
        vec1 = uv1d[:, self.vec2texImg_index]

        return vec1.reshape(self.batch_size, 3, -1)
        



class MANO_SMPL_HTML(MANO_SMPL):
    def __init__(self, mano_pkl_path, tex_model_path, uv_path):
        super(MANO_SMPL_HTML, self).__init__(mano_pkl_path, task_id=3)

        # need to read the face_uvs and verts_uvs.
        with open(tex_model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.tex_mean = self.model['mean']  # the mean texture
        self.tex_basis = self.model['basis']  # 101 PCA comps
        self.vec2texImg_index = self.model['index_map']  # the index map, from a compact vector to a 2D texture image

        with open(uv_path,'rb') as f:
            self.uvs = pickle.load(f)
        self.faces_uvs = self.uvs['faces_uvs']
        self.verts_uvs = self.uvs['verts_uvs']

        self.tex_num_pca_comp = 101
        self.tex_path = tex_model_path

        self.tex_img1d = torch.zeros(1024 * 1024 * 3)

        self.vec2texImg_index = torch.tensor(self.vec2texImg_index, dtype=torch.long).cuda()
        self.tex_basis = torch.tensor(self.tex_basis, dtype=torch.float).cuda()
        self.tex_mean = torch.tensor(self.tex_mean, dtype=torch.float).cuda()

        # mesh's face, uvs, etc
        self.faces_idx = torch.tensor(self.faces.astype(np.int) , dtype= torch.long).cuda()
        self.faces_uvs = torch.tensor(self.faces_uvs, dtype = torch.long).cuda()
        self.verts_uvs = torch.tensor(self.verts_uvs,dtype = torch.float).cuda()

    def vec2img(self, vec, vec2texImg_index):
        # super slow and inefficient. Reorganize the code vector
        batch_size = vec.shape[0]
        img1d = torch.zeros([batch_size, 1024 * 1024 * 3]).cuda()
        
        img1d[:, vec2texImg_index] = vec
        return img1d.reshape((batch_size, 3, 1024, 1024)).permute((0, 3, 2, 1))

    def get_mano_texture(self, gamma):
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, dtype=torch.float)
        if self.is_cuda:
            gamma.cuda()

        scaled_gamma = gamma.permute(1, 0)
        offsets = torch.mm(self.tex_basis, scaled_gamma)
        tex_code = offsets + self.tex_mean[..., None]
        new_tex_img = self.vec2img(tex_code.permute(1, 0), self.vec2texImg_index) / 255
        
        return new_tex_img



