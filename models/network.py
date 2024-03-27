import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import scipy.sparse as sp

import math
import numpy as np

from utils.utils import *

EPS = 1e-7
AUTOENCODER_FIX = True

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class PadSameConv2d(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)  
    

class PaddedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return t
    
class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)

class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=64, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()

        max_channels = 8 * nf
        num_layers = int(math.log2(in_size)) - 1
        channels = [cin] + [min(nf * (2 ** i), max_channels) for i in range(num_layers)]

        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1 if i != num_layers - 1 else 0, bias=False),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)]
        )
        if activation is not None:
            self.out_layer = nn.Sequential(nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False), activation())
        else:
            self.out_layer = nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x).reshape(x.size(0),-1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        
        return torch.bmm(adj, x)


class GCN(nn.Module):
    def __init__(self, node_features, hidden_dim, adj, in_size=256, use_bias=True):
        super(GCN, self).__init__()

        self.attentions = Encoder(cin=3, cout=778, in_size=in_size, activation=nn.Sigmoid)
        self.adj = self.normalize_adj(adj)
        self.gcn_1 = GCNLayer(node_features, hidden_dim, use_bias)
        self.gcn_2 = GCNLayer(hidden_dim, node_features, use_bias)

    def initialize_weights(self):
        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()

    def normalize_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)) # D
        d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        adj = torch.from_numpy(adj.toarray()).unsqueeze(0)
        adj = torch.tensor(adj, device='cuda', dtype=torch.float32)

        return adj
    

    def forward(self, render_diff, x):
        attentions = self.attentions(render_diff)
        
        x = x * attentions.unsqueeze(-1)
        x = F.relu(self.gcn_1(x, self.adj))
        x = F.tanh(self.gcn_2(x, self.adj))
        return x * 0.1
    

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, f_act=nn.Tanh):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=None, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=None, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, f_act=f_act)  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, f_act=nn.Tanh):

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if norm_layer is not None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)
            use_bias = False
        else:
            use_bias = True

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                kernel_size=3, stride=1,
                                padding=1, padding_mode='replicate')
            up = [uprelu, upsample, upconv]

            down = [downconv]
            if f_act is not None:
                up += [f_act()]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc,
                                kernel_size=1, stride=1,
                                padding=0, bias=use_bias, padding_mode='replicate')
            up = [uprelu, upsample, upconv]

            down = [downrelu, downconv]
            if norm_layer is not None:
                up += [upnorm]

            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                kernel_size=1, stride=1,
                                padding=0, bias=use_bias, padding_mode='replicate')
            up = [uprelu, upsample, upconv]

            down = [downrelu, downconv]
            if norm_layer is not None:
                down += [downnorm]
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

    
class BiTextureNet(nn.Module): 
    def __init__(self, input_nc=3, output_nc=3, oneD_size = 206330, mid_channel=[16, 64, 128, 256], batch_size=3, tex_size=512, f_act=nn.Tanh, device='cpu'):
        super(BiTextureNet, self).__init__()
        self.tex_size = tex_size
        self.batch_size = batch_size
        self.device = device
        
        self.encoding_channel = mid_channel
        self.pre_conv = ResnetBlock(input_nc, self.encoding_channel[0])
        
        self.pad_bin = []
        self.oneD_size = oneD_size
        D_size = self.oneD_size
        self.encs = nn.ModuleList([])
        self.decs = nn.ModuleList([])
        for num_down in range(len(self.encoding_channel)-1):
            self.encs.append(nn.ModuleList([
                ResnetBlock(self.encoding_channel[num_down], self.encoding_channel[num_down+1]),
                ResnetBlock(self.encoding_channel[num_down+1], self.encoding_channel[num_down+1]),
                Downsample(2),
            ]))
            if D_size % 2 == 0:
                self.pad_bin.insert(0, False)
            else:
                self.pad_bin.insert(0, True)
            D_size = D_size // 2


        self.mid_block1 = ResnetBlock(self.encoding_channel[-1], self.encoding_channel[-1])
        self.act = nn.ReLU()
        # self.attn = LinearAttention(encoding_channel[-1])
        self.mid_block2 = ResnetBlock(self.encoding_channel[-1], self.encoding_channel[-1])

        for num_down in range(len(self.encoding_channel)-2, -1, -1):
            self.decs.append(nn.ModuleList([
                ResnetBlock(self.encoding_channel[num_down+1]*3, self.encoding_channel[num_down]),
                ResnetBlock(self.encoding_channel[num_down], self.encoding_channel[num_down]),
                Upsample(2),
            ]))
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.encoding_channel[0], 8, 1),
            nn.Conv1d(8, output_nc, 1),
            nn.Sigmoid(),
        )

    def pad_upsample(self, vec):
        up_vec = self.upsample(vec)
        return  torch.cat([up_vec, up_vec[:, :, -1:]], dim=-1)
    
    def preprocess(self, oneD):
        return oneD.reshape(self.batch_size, 3, -1)
    
    # [B, 3, 206630]
    def forward(self, tex_1d, guide_tex_1d, visible_map_1d):
        
        tex_1d_l = tex_1d[0]
        tex_1d_r = tex_1d[1]
        guide_tex_1d_l = guide_tex_1d[0]
        guide_tex_1d_r = guide_tex_1d[1]

        tex_1d_l = self.preprocess(tex_1d_l)
        tex_1d_r = self.preprocess(tex_1d_r)
        guide_tex_1d_l = self.preprocess(guide_tex_1d_l)
        guide_tex_1d_r = self.preprocess(guide_tex_1d_r)
        visible_1d_l = self.preprocess(visible_map_1d[0])
        visible_1d_r = self.preprocess(visible_map_1d[1])
        
        l_board = tex_1d_l * visible_1d_l + guide_tex_1d_l * (1 - visible_1d_l)
        r_board = tex_1d_r * visible_1d_r + guide_tex_1d_r * (1 - visible_1d_r)
        
        l_board = self.pre_conv(l_board)
        r_board = self.pre_conv(r_board)
        # [206330, 103165, 51582, 25791, 12895, 6447, 3223, 1611]

        lx = l_board
        rx = r_board
        l_enc_feat = []
        r_enc_feat = []
        for block1, block2, downsample in self.encs:
            lx = block1(lx)  
            lx = block2(lx)
            lx = downsample(lx)
            l_enc_feat.append(lx)
            rx = block1(rx)
            rx = block2(rx)
            rx = downsample(rx)          
            r_enc_feat.append(rx)
            
        lx = self.mid_block1(lx)
        lx = self.mid_block2(lx)
        rx = self.mid_block1(rx)
        rx = self.mid_block2(rx)

        for i, (block1, block2, upsample) in enumerate(self.decs):
            l_init = lx
            r_init = rx
            
            lx = torch.cat([l_init, r_init, l_enc_feat.pop()], dim=1)
            lx = block1(lx)
            lx = upsample(lx)
            
            rx = torch.cat([r_init, l_init, r_enc_feat.pop()], dim=1)
            rx = block1(rx)
            rx = upsample(rx)
            
            if self.pad_bin[i]:
               lx = torch.cat([lx, lx[:, :, -1:]], dim=-1)
               rx = torch.cat([rx, rx[:, :, -1:]], dim=-1)
            
            lx = block2(lx)
            rx = block2(rx)

        l_final = self.final_conv(lx)
        r_final = self.final_conv(rx)

        return [l_final, r_final]
    
    
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data
    
def Upsample(scale=2):
    return nn.Upsample(scale_factor = scale, mode = 'nearest')

def pad_Upsample(vec, scale=2):
    up_vec  = Upsample(vec, scale)
    return torch.cat([up_vec, up_vec[:, :, -1:]], dim=-1)

def Downsample(scale):
    return nn.MaxPool1d(scale, scale)

def Downsample2D(scale):
    return nn.MaxPool2d(scale, scale)
 
def upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
    
    
    
class LinearAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm2D(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
                    
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        # import pd;
        q = q * self.scale

        # context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return q, k, v
    


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
    