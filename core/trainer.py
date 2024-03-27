import torch
from dataset.dataloader import *
from torch.utils.data import DataLoader
from core.loss import *

from tqdm import tqdm
from models.model import *

import time
from datetime import datetime

class Trainer:
    def __init__(self, model=None, cfgs=None, device='cpu'):
        self.cfgs = cfgs
        self.device = device

        # SIZE CONFIG
        self.img_size = cfgs.IMG_SIZE
        self.uv_size = cfgs.UV_SIZE

        # MODEL_PATH
        self.pretrain_path = cfgs.MODEL.PRETRAIN_PATH if cfgs.MODEL.PRETRAIN_PATH != 'none' else None

        # TRAIN CONFIG
        self.learning_rate = cfgs.TRAIN.LR
        self.batch_size = cfgs.TRAIN.BATCH_SIZE
        self.epochs = cfgs.TRAIN.EPOCHS
        self.decay_step = cfgs.TRAIN.lr_decay_step
        self.decay_gamma = cfgs.TRAIN.lr_decay_gamma 

        self.hand_type = cfgs.MODEL.HAND_TYPE

        self.refine_mask = cfgs.REFINE_MASK
        
        # DATASET
        self.limit = cfgs.DATASET.LIMIT       

        self.loss = RenderingLoss(cfgs)

        self.model = model
        if self.model is None:
            self.init_model()
        
        self.folder_path =os.path.join(cfgs.SAVE.FOLDER_PATH, cfgs.MODEL_TYPE + '_' + cfgs.DATASET.DATASET + '_'  + (datetime.now().strftime('%m-%d_%H-%M')))
        os.makedirs(self.folder_path, exist_ok=True)

        self.warmed_model = False
        self.running_img = None
        self.running_mask = None
        self.running_img_name = 'null'

    def load_data(self):
        if self.cfgs.DATASET.DATASET == 'InterHand':
            train_dataset =  InterHandTrainData(self.cfgs, limit=self.limit, set_type='train', device=self.device)
        elif self.cfgs.DATASET.DATASET == 'Rgb2Hands':
            train_dataset =  RGB2HandsTrainData(self.cfgs, limit=self.limit, set_type='train', device=self.device)
    
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle= True
        )

    def preprocess_data(self, data):
        data_dict = {}
        data_dict['img_name'] = data['image_name']
        data_dict['imgs'] = torch.stack(data['multiview_imgs'], dim=1).to(self.device)
        if 'multiview_hand' in data.keys():
            data_dict['hands'] = torch.stack(data['multiview_hand'], dim=1).to(self.device)
        if self.cfgs.MODEL.USE_GT_SHAPE:
            data_dict['anno'] = switch_dict( data['multiview_anno'] ) 
        if self.cfgs.TEST.NOVEL_POSE:
            data_dict['pose_imgs'] = torch.stack( data['pose_imgs'], dim=1).to(self.device) 
            if self.cfgs.MODEL.USE_GT_SHAPE:
                data_dict['pose_anno'] = switch_dict( data['pose_anno']) 
        
        return data_dict
    
    def sample_dict(self, data_dict, view_id=0):
        train_dict = {}
        
        # train_dict['img_name'] = data_dict['img_name']
        train_dict['imgs'] = data_dict['imgs'][view_id]
        train_dict['hands'] = data_dict['hands'][view_id]
        train_dict['anno'] = data_dict['anno'][view_id]
        
        return train_dict
        
    def init_model(self):
        self.model = BiTT(self.cfgs, hand_type=self.hand_type, device=self.device).cuda()
        
    def get_model(self):
        return self.model

    def load_model(self, part_lists = None):
        print('Loading model from', self.pretrain_path)
        pretrain_dict = torch.load(self.pretrain_path)
        model_dict = self.model.state_dict()
        if part_lists is None:
            self.model.load_state_dict(pretrain_dict)
        else:
            for each_p in part_lists:
                targets = {}
                for k, v in pretrain_dict.items():
                    if each_p in k:
                        targets['k'] = v

            self.model.load_state_dict(targets)

    def set_phase(self, img, mask, img_name=None):
        self.running_img = img
        self.running_mask = mask
        self.running_img_name = img_name


    def run_model(self, input_img, hand_img, hand_dict=None):
        self.warmed_model = True
        outputs = self.model(input_img, hand_img, hand_dict)

        return outputs

    def optimize(self, img=None, hand_img=None, img_name=None, hand_dict=None):
        img = self.running_img if img is None else img
        img_name = self.running_img_name if img_name is None else img_name
        
        best_lpips = 10000
        best_mss = 0
        best_psnr = 0

        init_epoch = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), 
                                        eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.decay_gamma, last_epoch = init_epoch-1)
        LEARNING_RATE_CLIP = 1e-5 
    
        pbar = tqdm(range(self.epochs), total=self.epochs, smoothing=0.9)
        for epoch in pbar:
            lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            
            total = 0
            seen = 0
            optimizer.zero_grad()           

            outputs = self.run_model(img, hand_img, hand_dict=hand_dict)
    
            loss = self.loss.ren_loss(outputs, overlap_mask=outputs['refine_mask'])

            total += loss['total'].item()
            seen += 1
            loss['total'].backward()

            optimizer.step()
                
            pbar.set_description("t:{:.4f}r:{:.4f}ps:{:.2f}lp:{:.3f}ms:{:.3f}".format(loss['total'].item(), loss['recon_loss'], loss['psnr_loss'], loss['lpips_loss'], loss['msssim_loss']))
            total /= seen
                    
            scheduler.step()
            if  (loss['msssim_loss'] > best_mss):
                best_mss = loss['msssim_loss'] 
                best_psnr = loss['psnr_loss'] 
                best_lpips = loss['lpips_loss'] 
                # torch.save(self.model.state_dict(), os.path.join(self.folder_path, '{}.pth'.format(img_name)))
            
        return outputs, best_mss, best_psnr, best_lpips

    def train_single(self, img, hand_img, img_name='null', hand_dict=None, train_full=False):
        self.set_phase(img, hand_img, img_name)
        
        self.img_folder_path = os.path.join(self.folder_path, img_name)
        os.makedirs(self.img_folder_path, exist_ok=True)
        self.model.train()

        outputs, best_mss, best_psnr, best_lpips = self.optimize(img, hand_img, img_name, hand_dict)

        print(f"Finished training {img_name}. mss: {best_mss}, psnr: {best_psnr}, lpips: {best_lpips}")

        return outputs

    def train_main(self, train=False):
        self.load_data()

        final_l1 = 0
        final_lpips = 0
        final_psnr = 0
        final_mss = 0
        final_ss = 0

        final_el1 = 0
        final_elpips = 0
        final_epsnr = 0
        final_emss = 0
        final_ess = 0

        final_nl1 = 0
        final_nlpips = 0
        final_npsnr = 0
        final_nmss = 0
        final_nss = 0
        
        dataset_info = {
            'left': 0,
            'right': 0,
            'symmetric': 0,
            'non_symmetric': 0,
            'full_occlusion': 0
        }
        
        for idx, (each_data) in enumerate(self.train_loader):
            print(f'Scene Number: {idx} of {len(self.train_loader)}')
            data_dict = self.preprocess_data(each_data)

            train_id = 0
            img_name = data_dict['img_name'][train_id].split('.')[0]
            input_im = data_dict['imgs'][:, train_id]
            input_hand = data_dict['hands'][:, train_id] if 'hands' in data_dict.keys() else None
            hand_dict =  each_data['multiview_anno'][train_id] if self.cfgs.MODEL.USE_GT_SHAPE else None
            
            self.init_model()
            self.img_folder_path = os.path.join(self.folder_path, img_name)
            os.makedirs(self.img_folder_path, exist_ok=True)    
            if train:
                outputs = self.train_single(input_im, input_hand, hand_dict=hand_dict,img_name=img_name)
            else:
                outputs = self.model.forward(input_im, input_hand, hand_dict)
            
            print(f"{idx} image name : {img_name}")
           
            data_summation = (outputs['visible_1d_mask'][0] +  outputs['visible_1d_mask'][1])
            dataset_info['left'] = dataset_info['left'] + torch.sum(outputs['visible_1d_mask'][0]) / (3 * 206330)
            dataset_info['right'] = dataset_info['right'] + torch.sum(outputs['visible_1d_mask'][1]) / (3 * 206330)
            dataset_info['non_symmetric'] = dataset_info['non_symmetric'] + torch.sum((data_summation == 2).int()) / (3 * 206330 * 2)
            dataset_info['symmetric'] = dataset_info['symmetric'] + torch.sum((data_summation == 1).int()) / (3* 206330 * 2) 
            dataset_info['full_occlusion'] = dataset_info['full_occlusion'] + torch.sum((data_summation == 0).int()) / (3 * 206330 * 2)
            
            input_hand, recon_hand = self.model.render_forward(input_im, outputs['recon_tex'], hand_dict=hand_dict)
            train_loss = self.loss.eval_loss(recon_hand, input_hand)
            self.save_results(outputs, method='recon', keys=['input_im', 'albedo_im', 'mesh_visual', 'alb_recon_hand', 'recon_hand', 'coarse_hand', 'recon_hand_new', 'rl_recon_alb_im', 'rl_recon_alb_im2', 'recon_hand_rl', 'recon_hand_rl2'])

            final_l1        += train_loss['recon_loss']
            final_lpips     += train_loss['lpips_loss']
            final_psnr      += train_loss['psnr_loss']
            final_mss       += train_loss['msssim_loss']
            final_ss        += train_loss['ssim_loss']
                 
            ## Evaluation      
            with torch.no_grad():
                ## Novel View Evaluation
                if self.cfgs.TEST.NOVEL_VIEW:
                    N_VIEW = len(each_data['multiview_imgs'])
                    batch_input_view = []
                    batch_novel_view = []
                    for i in range(1, N_VIEW):
                        input_im = data_dict['imgs'][:, i]
                        input_hand = data_dict['hands'][:, i]
                        hand_dict = each_data['multiview_anno'][i]

                        input_hand, recon_hand = self.model.render_forward(input_im, outputs['recon_tex'], hand_dict)
                        batch_input_view.append(input_hand)
                        batch_novel_view.append(recon_hand)
                    batch_input_view = torch.stack(batch_input_view, dim=1).to(self.device)
                    batch_novel_view = torch.stack(batch_novel_view, dim=1).to(self.device)
                    
                    nv_eval_loss = self.loss.eval_loss(batch_novel_view[0], batch_input_view[0])
                    self.save_batch_view(batch_novel_view, method='recon')
                    self.save_batch_view(data_dict['hands'], method='input')
                    
                    final_el1        += nv_eval_loss['recon_loss']
                    final_elpips     += nv_eval_loss['lpips_loss']
                    final_epsnr      += nv_eval_loss['psnr_loss']
                    final_emss       += nv_eval_loss['msssim_loss']
                    final_ess        += nv_eval_loss['ssim_loss']
                    
                    print(f"[NV_Evaluation] mss: {nv_eval_loss['msssim_loss'] }, psnr: {nv_eval_loss['psnr_loss'] }, lpips: {nv_eval_loss['lpips_loss'] }")
                
                ## Novel Pose Evaluation
                if self.cfgs.TEST.NOVEL_POSE:
                    N_POSE = len(each_data['pose_imgs'])
                    batch_input_pose = []
                    batch_novel_pose = []
                    for i in range(0, N_POSE):
                        input_im = data_dict['pose_imgs'][:, i]

                        hand_dict = each_data['pose_anno'][i] if self.cfgs.MODEL.USE_GT_SHAPE else None
                        input_hand, recon_hand = self.model.render_forward(input_im, outputs['recon_tex'], hand_dict)
            
                        batch_input_pose.append(input_hand)
                        batch_novel_pose.append(recon_hand)
                    batch_input_pose = torch.stack(batch_input_pose, dim=1).to(self.device)
                    batch_novel_pose = torch.stack(batch_novel_pose, dim=1).to(self.device)

                    np_eval_loss = self.loss.eval_loss(batch_novel_pose[0], batch_input_pose[0])
                    self.save_batch_view(batch_novel_pose, method='novel_pose')
                    self.save_batch_view(data_dict['pose_imgs'], method='input_pose')

                    final_nl1        += np_eval_loss['recon_loss']
                    final_nlpips     += np_eval_loss['lpips_loss']
                    final_npsnr      += np_eval_loss['psnr_loss']
                    final_nmss       += np_eval_loss['msssim_loss']
                    final_nss        += np_eval_loss['ssim_loss']


            print(f"[NP_Evaluation] mss: {np_eval_loss['msssim_loss'] }, psnr: {np_eval_loss['psnr_loss'] }, lpips: {np_eval_loss['lpips_loss']}")

            del self.model
            
        final_l1    /= len(self.train_loader)
        final_lpips /= len(self.train_loader)
        final_psnr  /= len(self.train_loader)
        final_ss    /= len(self.train_loader)
        final_mss   /= len(self.train_loader)
            
        final_el1   /= len(self.train_loader)
        final_elpips/= len(self.train_loader)
        final_epsnr /= len(self.train_loader)
        final_ess   /= len(self.train_loader)
        final_emss  /= len(self.train_loader)
            
        final_nl1   /= len(self.train_loader)
        final_nlpips/= len(self.train_loader)
        final_npsnr /= len(self.train_loader)
        final_nss   /= len(self.train_loader)
        final_nmss  /= len(self.train_loader)
        
        data_info_result = 'Left: {} \n \
            Right: {}\n \
            Symmetric: {}\n \
            Non Symmetric: {}\n \
            Occluded: {} '.format(dataset_info['left'] / len(self.train_loader), dataset_info['right'] / len(self.train_loader), dataset_info['symmetric'] / len(self.train_loader), dataset_info['non_symmetric'] / len(self.train_loader), dataset_info['full_occlusion'] / len(self.train_loader))
        print(data_info_result)
        

        result = 'Recon) l1:{}, lpips:{}, psnr:{}, ss:{}, mss:{}\n  \
                NV) l1:{}, lpips:{}, psnr:{}, ss:{}, mss:{}\n  \
                NP) l1:{}, lpips:{}, psnr:{}, ss:{}, mss:{}\n'.format(
            final_l1, final_lpips, final_psnr, final_ss, final_mss,
            final_el1, final_elpips, final_epsnr, final_ess, final_emss,
            final_nl1, final_nlpips, final_npsnr, final_nss, final_nmss,
        )
        print(result)
        f = open(os.path.join(self.folder_path, 'result.txt'), 'w')
        f.write(result)
        f.close()
    
    def changeView(self, tex_list, cam_view = [0, -0.5, 0, 1]):
        if self.warmed_model is False:
            self.run_model(self.running_img, self.running_mask)
        with torch.no_grad():
            result = self.model.novel_view(tex_list=tex_list, camera_direction=cam_view)

        return result
    
    def changePose(self, target_pose_im):
        if self.warmed_model is False:
            self.run_model(self.running_img, self.running_mask)
        with torch.no_grad():
            datas = self.model.change_pose_by_ref(target_pose_im, return_mask=True)
        return datas

    def eval(self, img, mask=None, hand_dict=None, img_name='null' ,save=True):
        self.model.eval()
        # Novel View Eval
        sT = time.time()
        
        with torch.no_grad():
            eval_results = self.model.test_forward(img, mask, hand_dict)
        print('Inference Time: {}'.format(time.time() - sT))
        
        loss_dict = self.loss.intersection_eval_loss(eval_results['recon_hand'], eval_results['input_hand'], mask, eval_results['mesh_mask'])
    
        return eval_results, loss_dict
    
    def save_results(self, results, method='recon', keys=['input_im', 'coarse_hand', 'recon_hand', 'mesh_mask', 'recon_hand_new', 'recon_hand_rl', 'depth_map', 'normal_map']):
        save_img(self.img_folder_path, self.running_img_name, results, keys=keys, method=method)
    
    def save_batch_view(self, batch_view, method='recon'):
        n_view = batch_view.size(1)
        for v in range(n_view):
            cv2.imwrite(os.path.join(self.img_folder_path, f'{method}_{v}.jpg'), tensor2img(batch_view[0][v].unsqueeze(0), as_cv=True))