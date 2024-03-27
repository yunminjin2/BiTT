import os
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_cfg
from utils.utils import *
from core.trainer import Trainer
from models.model import *

from dataset.dataloader import *

def custom_load_mano(data, mano_layer):
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
        handV, handJ = mano_layer[hand_type](torch.from_numpy(params['R']).float(), torch.from_numpy(params['pose']).float(), torch.from_numpy(params['shape']).float(), trans=torch.from_numpy(params['trans']).float())
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




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom', action='store_true')
    parser.add_argument("--cfg", type=str, default=os.path.join('misc', 'model', 'config.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test',  action='store_true')
    
    parser.add_argument('--img', default='demo/TwoHand/IMG_6060.png', type=str, help='Specify a image')
    parser.add_argument('--hand_dict', type=str, help='Specify a image')
    parser.add_argument('--mask', type=str, help='Specify a image')
    parser.add_argument('--pose', default='demo/Pose/219617.jpg', type=str, help='Specify a image')
    parser.add_argument('--pose_dict', type=str, help='Specify a image')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt) 
    
    
    trainer = Trainer(
        cfgs=cfg,
        device=device
    )
    
    if opt.custom: 
        hand_dict = None
        
        img_path  = opt.img
        img_name = os.path.basename(img_path).split('.')[0]
        
        mask_path = img_path if opt.mask is None else opt.mask
        img =  cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        imgT = img2tensor(img, img_size=cfg.IMG_SIZE, device=device, crop=False).unsqueeze(0)
        
        maskT = img2tensor(mask, img_size=cfg.IMG_SIZE, device=device, crop=False).unsqueeze(0)
        maskT = (maskT > 0.1).int()
        trainer.set_phase(imgT, maskT, img_name)

        if opt.hand_dict:
            mano_base = cfg.MISC.MANO_PATH
            mano_path = {'left': os.path.join(mano_base, 'MANO_LEFT.pkl'), 'right': os.path.join(mano_base, 'MANO_RIGHT.pkl')}
            mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                            'left': ManoLayer(mano_path['left'], center_idx=None)}
            fix_shape(mano_layer)
            renderer = mano_two_hands_renderer(img_size=cfg.IMG_SIZE, mano_path=mano_path, device=device)
    
            # dataset = InterHandTrainData(cfg)            
            
            anno_path = opt.hand_dict
            with open(anno_path, 'rb') as file:
                data = pickle.load(file)

            hand_dict = custom_load_mano(data, mano_layer)
            hand_dict['left'] = dict2tensor(hand_dict['left'], device=device)
            hand_dict['right'] = dict2tensor(hand_dict['right'], device=device)
            
        if opt.train:
            print("Training the Model")
            
            outputs = trainer.train_single(imgT, maskT, img_name, hand_dict=hand_dict)
            
            trainer.save_results(outputs, method='recon', keys=['input_im', 'albedo_im', 'recon_hand', 'guide_hand', 'recon_hand_new', 'recon_hand_rl', 'alb_recon_hand', 'refine_mask', 'mesh_visual'])
        if opt.test:
            # dataset = InterHandTrainData(cfg)            
            pose_hand_dict = None
            if opt.pose_dict:
                anno_path = opt.pose_dict
                with open(anno_path, 'rb') as file:
                    data = pickle.load(file)

                pose_hand_dict = custom_load_mano(data, mano_layer)
                pose_hand_dict['left'] = dict2tensor(pose_hand_dict['left'], device=device)
                pose_hand_dict['right'] = dict2tensor(pose_hand_dict['right'], device=device)
                
            pimg =  cv2.imread(opt.pose)
            pimgT = img2tensor(pimg, img_size=cfg.IMG_SIZE, device=device, crop=False).unsqueeze(0)
            train_outputs = trainer.run_model(imgT, maskT, hand_dict)
            model = trainer.get_model()
            input_hand, recon_hand = model.render_forward(pimgT, train_outputs['recon_tex'], hand_dict=pose_hand_dict)
            trainer.save_results({'recon_hand':recon_hand}, method='np', keys=['recon_hand'])    
            
            trainer.save_results(train_outputs, method='recon', keys=['input_im', 'albedo_im', 'recon_hand', 'guide_hand', 'recon_hand_new', 'recon_hand_rl', 'recon_hand_rl2', 'alb_recon_hand', 'refine_mask', 'mesh_visual'])
            results = trainer.changeView(train_outputs['recon_tex'], [0, 0, 0, -1])
            trainer.save_results(results, method='nv', keys=['recon_hand', 'guide_hand'])
    else:
        trainer.train_main(train=True)