import numpy as np
import random
import math
import cv2
import pickle
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset_utils import IMG_SIZE
from utils.config import get_cfg_defaults
from models.model_zoo import build_graph
from numpy.core.numeric import zeros_like


def save_video(out_fold, frames, fname='image', ext='.mp4', cycle=False):
    os.makedirs(out_fold, exist_ok=True)
  # TxCxHxW -> TxHxWxC
    if cycle:
        frames = np.concatenate([frames, frames[::-1]], 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vid = cv2.VideoWriter(os.path.join(out_fold, fname+ext), fourcc, 60, (frames.shape[2], frames.shape[1]))
    [vid.write(f) for f in frames]
    vid.release()

def save_img(out_fold, img_name, outputs, keys=['input_im', 'recon_im'], method='recon'):
    for src in keys:
        try:
            cv2.imwrite(os.path.join(out_fold, f'{img_name}_{method}_{src}.jpg'), tensor2img(outputs[src], as_cv=True))
        except:
            continue
def lightenUVmap(uv_map): # [B, 3, 1024, 1024]
    return uv_map[:, :, 1024 - 768:, :]

def restoreUVmap(uv_map): # [B,3 512, 1024]:
    B, C, H, W = uv_map.size()
    back = torch.zeros([B, C, 1024 - H, W]).to(uv_map.device)
    return torch.cat([back, uv_map], dim=2)

def switch_dict(this_dict):
    batch_res = []
    batch_size = len(this_dict[0]['left']['hand_type'])
    for each_batch in range(batch_size):
        tmp = {
            'left': {
                'camera':[],
                'verts3d': []
            },
            'right': {
                'verts3d':[]
            }
        }
        for each_view_dict in this_dict:
            # tmp['hand_type'] = each_view_dict['hand_type'][each_batch]
            
            tmp['left']['camera'].append(each_view_dict['left']['camera'][each_batch])
            tmp['left']['verts3d'].append(each_view_dict['left']['verts3d'][each_batch])
            tmp['right']['verts3d'].append(each_view_dict['right']['verts3d'][each_batch])
            
        batch_res.append(tmp)
            
    return batch_res

def grabcut_refine(mano_mask, rgb_img):
    rgb_img = rgb_img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255.0
    rgb_img = rgb_img.astype(np.uint8)
    #erode the MANO mask
    
    kernel = np.ones((25,25),np.uint8)
    mano_mask_eroded = cv2.erode(mano_mask*255, kernel, iterations=1)
        
    grabCut_mask = zeros_like(mano_mask)
    grabCut_mask[mano_mask_eroded > 0] = cv2.GC_PR_FGD
    grabCut_mask[mano_mask_eroded == 0] = cv2.GC_PR_BGD

    #GRABCUT
    # allocate memory for two arrays that the GrabCut algorithm internally uses when segmenting the foreground from the background
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    # apply GrabCut using the the mask segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(rgb_img, grabCut_mask, None, bgModel, fgModel, iterCount=20, mode=cv2.GC_INIT_WITH_MASK)

    # set all definite background and probable background pixels to 0 while definite foreground and probable foreground pixels are set to 1, then scale teh mask from the range [0, 1] to [0, 255]
    refined_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD) , 0, 1)
    refined_mask = (refined_mask * 255).astype("uint8")
    refined_mask = refined_mask[..., 0]
    
    return refined_mask

def largest_component(mask):
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    # import pdb; pdb.set_trace()
    
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])       # Note: range() starts from 1 since 0 is the background label.
    finalMask = zeros_like(mask)
    finalMask[labels == max_label] = 255
    return finalMask


def remove_forearm(mano_mask, mano_mask_refined):
    
    kernel = np.ones((10,10),np.uint8)
    mano_mask_dilated = cv2.dilate(mano_mask, kernel, iterations=1)
    _, diff = cv2.threshold(mano_mask_refined - mano_mask_dilated, 127, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(diff) == 0:         #mano_mask_dilated encapsulates the mano_mask_refined; nothing to remove
        return mano_mask_refined
    
    probable_forearm = largest_component(diff)
    #estimate mask area
    mask_area_frac = cv2.countNonZero(probable_forearm)/(mano_mask.shape[0]*mano_mask.shape[1])
    
    if mask_area_frac > 0.01:
        #extra region big enough to be a forearm
        return mano_mask_refined - probable_forearm
    else:
        #its probably some part of the palm
        return mano_mask_refined
        
def mask_refinement(img, mask, device):
    mask_1 = mask[:, 0:1]
    mano_mask_np = mask_1.permute(0, 2, 3, 1).cpu().detach().numpy()[0].astype(np.uint8)
    mano_mask_gc = grabcut_refine(mano_mask_np, img)
    
    if mano_mask_gc.max() != 0:
        mano_mask_gc_lc = largest_component(mano_mask_gc)
        mano_mask_forearm_removed = remove_forearm(mano_mask_np, mano_mask_gc_lc)
    
        refined_mask = torch.from_numpy(mano_mask_forearm_removed).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1 ,1).to(device)
        mask = ((mask - refined_mask) > 0.5).int().to(device)
    else:
        print('Cannot Refine a mask!')
    return mask

def tensor2img(input_tensor, batch_id=0, as_cv=False): # input_tensor [B, 3(1), H, W]
    if len(input_tensor.size()) == 4:
        input_tensor = input_tensor[batch_id]
    if input_tensor.size(0) == 1: # Mask img
        input_tensor = input_tensor.repeat(3, 1, 1)
    img = input_tensor.permute(1, 2, 0).detach().cpu().numpy()
    if not (input_tensor.max() > 1):
        img *= 255
    img = img.astype(np.uint8)
    
    return  img[..., ::-1] if as_cv else img

def img2tensor(img, img_size=IMG_SIZE, crop=False, device='cpu'):
    img = imgUtils.cut2squre(img) if crop else imgUtils.pad2squre(img)
    img = cv2.resize(img, (img_size, img_size))
    imgTensor = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32) / 255
    imgTensor = imgTensor.permute(2, 0, 1).to(device)
    
    return imgTensor

def dict2tensor(some_dict, device='cpu'):
    result = {}
    # import pdb; pdb.set_trace()
    for each_key, each_value in some_dict.items():
        if type(each_value) == str:
            result[each_key] = [each_value]
        else:
            result[each_key] = torch.tensor(each_value, device=device).unsqueeze(0)
    return result

def setup_runtime(args, seed=44, num_workers=4):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu

    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'


    print(f"Environment: GPU {device} seed {seed} number of workers {num_workers}")

    return device





def projection(scale, trans2d, label3d, img_size=256):
    scale = scale * img_size
    trans2d = trans2d * img_size / 2 + img_size / 2
    trans2d = trans2d

    label2d = scale * label3d[:, :2] + trans2d
    return label2d


def projection_batch(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def projection_batch_np(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale[..., np.newaxis, np.newaxis]
    if scale.dim() == 2:
        scale = scale[..., np.newaxis]
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d[:, np.newaxis, :]

    label2d = scale * label3d[..., :2] + trans2d
    return label2d

def get_model_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tex_path = os.path.join(abspath, cfg.MISC.PRE_TRAIN_PATH)
    return tex_path

def get_mano_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.MANO_PATH)
    mano_path = {'left': os.path.join(path, 'MANO_LEFT.pkl'),
                 'right': os.path.join(path, 'MANO_RIGHT.pkl')}
    return mano_path

def get_tex_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tex_path = os.path.join(abspath, cfg.MISC.TEX_PATH)
    return tex_path

def get_uvs_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.UVS_PATH)
    tex_path = {'left': os.path.join(path, 'uvs_left.pkl'),
                 'right': os.path.join(path, 'uvs_right.pkl')}
    return tex_path


def get_graph_dict_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graph_path = {'left': os.path.join(abspath, cfg.MISC.GRAPH_LEFT_DICT_PATH),
                  'right': os.path.join(abspath, cfg.MISC.GRAPH_RIGHT_DICT_PATH)}
    return graph_path


def get_dense_color_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dense_path = os.path.join(abspath, cfg.MISC.DENSE_COLOR)
    return dense_path


def get_mano_seg_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    seg_path = os.path.join(abspath, cfg.MISC.MANO_SEG_PATH)
    return seg_path


def get_upsample_path(cfg=None):
    if cfg is None:
        cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    upsample_path = os.path.join(abspath, cfg.MISC.UPSAMPLE_PATH)
    return upsample_path


def build_mano_graph(cfg=None):
    graph_path = get_graph_dict_path(cfg)
    mano_path = get_mano_path(cfg)
    for hand_type in ['left', 'right']:
        if not os.path.exists(graph_path[hand_type]):
            manoData = pickle.load(open(mano_path[hand_type], 'rb'), encoding='latin1')
            faces = manoData['f']
            graph_dict = build_graph(faces, coarsening_levels=4)
            with open(graph_path[hand_type], 'wb') as file:
                pickle.dump(graph_dict, file)


class imgUtils():
    @ staticmethod
    def pad2squre(img, color=None, crop=False):
        if img.shape[0] > img.shape[1]:
            W = img.shape[0] - img.shape[1]
        else:
            W = img.shape[1] - img.shape[0]
        W1 = int(W / 2)
        W2 = W - W1

        if crop:
            h = img.shape[0]
            w = img.shape[1]
            result = img[h//2 - w//2: h//2 + w//2]
            return result
        else:
            if color is None:
                if img.shape[2] == 3:
                    color = (0, 0, 0)
                else:
                    color = 0
            if img.shape[0] > img.shape[1]:
                return cv2.copyMakeBorder(img, 0, 0, W1, W2, cv2.BORDER_CONSTANT, value=color)
            else:
                return cv2.copyMakeBorder(img, W1, W2, 0, 0, cv2.BORDER_CONSTANT, value=color)

    @ staticmethod
    def cut2squre(img):
        if img.shape[0] > img.shape[1]:
            s = int((img.shape[0] - img.shape[1]) / 2)
            return img[s:(s + img.shape[1])]
        else:
            s = int((img.shape[1] - img.shape[0]) / 2)
            return img[:, s:(s + img.shape[0])]

    @ staticmethod
    def get_scale_mat(center, scale=1.0):
        scaleMat = np.zeros((3, 3), dtype='float32')
        scaleMat[0, 0] = scale
        scaleMat[1, 1] = scale
        scaleMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - scaleMat), center)
        scaleMat[0, 2] = t[0]
        scaleMat[1, 2] = t[1]
        return scaleMat

    @ staticmethod
    def get_rotation_mat(center, theta=0):
        t = theta * (3.14159 / 180)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - rotationMat), center)
        rotationMat[0, 2] = t[0]
        rotationMat[1, 2] = t[1]
        return rotationMat

    @ staticmethod
    def get_rotation_mat3d(theta=0):
        t = theta * (3.14159 / 180)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        return rotationMat

    @ staticmethod
    def get_affine_mat(theta=0, scale=1.0,
                       u=0, v=0,
                       height=480, width=640):
        center = np.array([width / 2, height / 2, 1], dtype='float32')
        rotationMat = imgUtils.get_rotation_mat(center, theta)
        scaleMat = imgUtils.get_scale_mat(center, scale)
        trans = np.identity(3, dtype='float32')
        trans[0, 2] = u
        trans[1, 2] = v
        affineMat = np.matmul(scaleMat, rotationMat)
        affineMat = np.matmul(trans, affineMat)
        return affineMat

    @staticmethod
    def img_trans(theta, scale, u, v, img):
        size = img.shape[0]
        u = int(u * size / 2)
        v = int(v * size / 2)
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=256, width=256)
        return cv2.warpAffine(src=img,
                             M=affineMat[0:2, :],
                             dsize=(256, 256),
                             dst=img,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE,
                             borderValue=(0, 0, 0)
                             )

    @staticmethod
    def data_augmentation(theta, scale, u, v,
                          img_list=None, label2d_list=None, label3d_list=None,
                          R=None,
                          img_size=224):
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=img_size, width=img_size)
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(cv2.warpAffine(src=img,
                                                  M=affineMat[0:2, :],
                                                  dsize=(img_size, img_size)))
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_list_out.append(np.matmul(label2d, affineMat[0:2, 0:2].T) + affineMat[0:2, 2:3].T)
        else:
            label2d_list_out = None

        if label3d_list is not None:
            label3d_list_out = []
            R_delta = imgUtils.get_rotation_mat3d(theta)
            for label3d in label3d_list:
                label3d_list_out.append(np.matmul(label3d, R_delta.T))
        else:
            label3d_list_out = None

        if R is not None:
            R_delta = imgUtils.get_rotation_mat3d(theta)
            R = np.matmul(R_delta, R)
        else:
            R = None

        return img_list_out, label2d_list_out, label3d_list_out, R

    @ staticmethod
    def add_noise(img, noise=0.00, scale=255.0, alpha=0.3, beta=0.05):
        # add brightness noise & add random gaussian noise
        a = np.random.uniform(1 - alpha, 1 + alpha, 3)
        b = scale * beta * (2 * random.random() - 1)
        img = a * img + b + scale * np.random.normal(loc=0.0, scale=noise, size=img.shape)
        img = np.clip(img, 0, scale)
        return img




def gaussian_noise_filetering_1d(noisy_1d, kenel=3):
    def gaussian_function(x, sigma):
        return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x**2) / (2 * sigma**2))

    def create_gaussian_filter(size, sigma):
        # Create an array of integers from -(size//2) to size//2
        x = np.arange(-(size // 2), (size // 2) + 1)
        # Compute the Gaussian function for each value in the array
        g_filter = gaussian_function(x, sigma)
        # Normalize the filter to ensure its sum equals 1
        g_filter /= np.sum(g_filter)
        return g_filter

    def apply_gaussian_filter(data, filter):
        return np.convolve(data, filter, mode='same')
    
    
    gaussian_filter = create_gaussian_filter(kenel, 2)  

    smoothed_data = apply_gaussian_filter(noisy_1d[0], gaussian_filter)    

    return smoothed_data