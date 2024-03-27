from torch.utils.data import Dataset
from utils.vis_utils import mano_two_hands_renderer
from models.manolayer import ManoLayer
from utils.utils import *
import numpy as np
import random
from dataset.dataset_utils import *

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


class InterHandTrainData(Dataset):
    def __init__(self, cfg, set_type='train', device='cpu', limit=1e6):
        super(InterHandTrainData, self).__init__()
        self.img_size = cfg.IMG_SIZE
        self.uv_size = cfg.UV_SIZE
        self.device = device
        
        base_path = cfg.DATASET.INTERHAND_PATH
        self.data_path = os.path.join(base_path, set_type)

        self.n_view = cfg.N_VIEW
        self.data_info_dict = {} 
        
        if limit == -1:
            limit=1e6
        
        target_cam_ids = os.listdir(os.path.join(self.data_path, 'masked_img_ref'))
        self.data_info_dict = {}
        for each_id in range(26):
            each_id = str(each_id)
            self.data_info_dict[each_id] = {}
            for target_cam in target_cam_ids:
                if len(self.data_info_dict[each_id].keys()) > limit:
                    break
                img_lists = os.listdir(os.path.join(self.data_path, 'masked_img_ref', target_cam))
                # import pdb; pdb.set_trace()
                for each_img in img_lists:
                    each_file = each_img.split('.')[0]
                    self.data_info_dict[each_id][each_file] = {
                        'cam_views': [target_cam],
                    }
                    for each_view in os.listdir(os.path.join(self.data_path, 'img', each_id)):
                        each_view_cam_ids = os.listdir(os.path.join(self.data_path, 'img', each_id, each_view))
                        if each_img in each_view_cam_ids:
                            self.data_info_dict[each_id][each_file]['cam_views'].append(each_view) 
                        if len(self.data_info_dict[each_id][each_file]['cam_views']) >= self.n_view:
                            break
        self.subdivide_multiview(self.n_view)
        
        ## MANO Loader
        mano_base = cfg.MISC.MANO_PATH
        mano_path = {'left': os.path.join(mano_base, 'MANO_LEFT.pkl'), 'right': os.path.join(mano_base, 'MANO_RIGHT.pkl')}
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)
        self.renderer = mano_two_hands_renderer(img_size=self.img_size, mano_path=mano_path, device=self.device)
    
    def subdivide_multiview(self, multi_view_size):
        self.data_info_list = []
        
        self.data_info_copy = {}
        for each_id, each_id_dict in self.data_info_dict.items():
            self.data_info_copy[each_id] = {}
            seq_list = each_id_dict.keys() 
            for each_im, each_dict in each_id_dict.items():
                if len(each_dict['cam_views']) < multi_view_size:
                    continue
                else:
                    self.data_info_copy[each_id][each_im] = {
                        'cam_views': each_dict['cam_views'],
                    }
            
        for each_id, each_id_dict in self.data_info_copy.items():
            seq_list = each_id_dict.keys() 
            for each_im, each_dict in each_id_dict.items():
                
                views = each_dict['cam_views'][:multi_view_size]
                pose = random.sample(seq_list, min(8, len(seq_list)))
                
                sub_dict = {'image_name': each_im, 
                            'id': str(each_id),
                            'cam_views':views, 
                            'test_poses': {
                                'image_name': pose,
                                'cam_view': [each_id_dict[each_im]['cam_views'][0] for each_im in pose]
                            }
                        }
                self.data_info_list.append(sub_dict)
        
    def load_mano(self, data):
        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']
        hand_dict = {}
        single_hand = False
        for idx, hand_type in enumerate(['left', 'right']):
            if data['mano_params'][hand_type] is None:
                h = ['left', 'right']
                single_hand = h[1 - idx]
                continue
            params = data['mano_params'][hand_type]
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(), torch.from_numpy(params['pose']).float(), torch.from_numpy(params['shape']).float(), trans=torch.from_numpy(params['trans']).float())
            handV = handV[0].numpy()
            handJ = handJ[0].numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            hand_dict[hand_type] = {'verts3d': handV, 
                                    'camera': camera,
                                    'hand_type':hand_type
                                    }
    
        return hand_dict
    
    def process_mask(self, mask):
        mask = torch.sum(mask, dim=0).unsqueeze(0)
        mask = (mask > 0.1).int()
        
        if mask.size(0) == 1:
            mask = mask.repeat(3, 1, 1)
        return mask
        
    def __getitem__(self, idx):
        data_dict = {}
        data_info = self.data_info_list[idx]
        image_name = data_info['image_name']
        capture_id = data_info['id']

        multiview_imgs = []
        multiview_hand = []
        multiview_anno = []
        test_hands = []
        test_anno = []

        for each_view in data_info['cam_views']:
            train_img_path = os.path.join(self.data_path, 'img', capture_id, each_view, image_name + '.jpg')
            hand_img_path = os.path.join(self.data_path, 'masked_img', capture_id, each_view, image_name + '.jpg')
            anno_path = os.path.join(self.data_path, 'anno', capture_id, each_view, image_name + '.pkl')

            img  = cv2.imread(train_img_path)
            hand = cv2.imread(hand_img_path)
            if img is None:
                import pdb; pdb.set_trace()
            img = img2tensor(img, img_size=self.img_size, crop=False, device=self.device)
            hand = img2tensor(hand, img_size=self.img_size, crop=False, device=self.device)
            
            with open(anno_path, 'rb') as file:
                data = pickle.load(file)
           
            hand_dict = self.load_mano(data)
                      
            multiview_imgs.append(img)
            multiview_hand.append(hand)
            multiview_anno.append(hand_dict)
        
        for each_pose in range(len(data_info['test_poses']['image_name'])):
            image_name = data_info['test_poses']['image_name'][each_pose]
            view = data_info['test_poses']['cam_view'][each_pose]

            train_img_path = os.path.join(self.data_path, 'img', capture_id, view, image_name + '.jpg')
            hand_img_path = os.path.join(self.data_path, 'masked_img', capture_id, view, image_name + '.jpg')
            anno_path = os.path.join(self.data_path, 'anno', capture_id, view, image_name + '.pkl')

            img  = cv2.imread(train_img_path)
            hand = cv2.imread(hand_img_path)
            
            img = img2tensor(img, img_size=self.img_size, crop=False, device=self.device)
            hand = img2tensor(hand, img_size=self.img_size, crop=False, device=self.device)
            
            with open(anno_path, 'rb') as file:
                data = pickle.load(file)
           
            hand_dict = self.load_mano(data)
                      
            test_hands.append(hand)
            test_anno.append(hand_dict)



        data_dict['image_name'] = image_name
        data_dict['multiview_imgs'] = multiview_imgs
        data_dict['multiview_hand'] = multiview_hand
        data_dict['multiview_anno'] = multiview_anno
        data_dict['pose_imgs'] = test_hands
        data_dict['pose_anno'] = test_anno
    
        return data_dict
    
    def __len__(self):
        return len(self.data_info_list)

class RGB2HandsTrainData(Dataset):
    def __init__(self, cfg, set_type='train', device='cpu', limit=3e2):
        super(RGB2HandsTrainData, self).__init__()
        self.img_size = cfg.IMG_SIZE
        self.uv_size = cfg.UV_SIZE
        self.device = device
        
        self.data_path = cfg.DATASET.RGB2HANDS_PATH

        self.n_view = cfg.N_VIEW
        self.data_info_dict = {} 
        
        if limit == -1:
            limit=int(3e2)
        
        self.img_ls = os.listdir(os.path.join(self.data_path, 'imgs'))

        # make group
        # 0 ~ 312, 313 ~ 762, 763 ~ 1331, 1332 ~ 
        self.training_imgs =  []
        for i in range(limit):
            rand_id = random.randint(0, len(self.img_ls) - 1)
            
            self.training_imgs.append(self.img_ls[rand_id])
        

    def process_mask(self, mask):
        mask = torch.sum(mask, dim=0).unsqueeze(0)
        mask = (mask > 0.1).int()
        
        if mask.size(0) == 1:
            mask = mask.repeat(3, 1, 1)
        return mask
  
    
    def __getitem__(self, idx):
        data_dict = {}
        image_name = self.training_imgs[idx]
        img_id = image_name.split('.')[0]
        
        img_id = int(img_id.split('_')[0])
                
        test_hands = []
        test_anno = []
        
        # get group
        img_range = np.arange(0, 313)
        group = 0
        if 312 < img_id and img_id < 763:
            img_range = np.arange(313, 763)
            group = 1
        elif 762 < img_id and img_id < 1332:
            img_range = np.arange(763, 1332)
            group = 2 
        elif 1331 < img_id:
            img_range = np.arange(1332, len(self.img_ls))
            group = 3
            
    
        img = cv2.imread(os.path.join(self.data_path, 'imgs', image_name))
        hands = cv2.imread(os.path.join(self.data_path, 'masked_imgs', image_name))
        
        img = img2tensor(img, img_size=self.img_size, crop=False, device=self.device)
        hands = img2tensor(hands, img_size=self.img_size, crop=False, device=self.device)
        test_imgs = []
        
        for each_pose in range(self.n_view):
            idx = np.random.choice(img_range)
            test_img = cv2.imread(os.path.join(self.data_path, 'imgs', '{:07}_output.jpg'.format(idx)))
            test_img_tensor = img2tensor(test_img, img_size=self.img_size, crop=False, device= self.device)
            test_imgs.append(test_img_tensor)
        data_dict['image_name'] = image_name
        data_dict['multiview_imgs'] = [img]
        data_dict['multiview_hand'] = [hands]
        data_dict['pose_imgs'] = test_imgs
        
        return data_dict
    
    def __len__(self):
        return len(self.training_imgs)

