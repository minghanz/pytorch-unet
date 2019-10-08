
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import networks

from dataloader import ImgPoseDataset, ToTensor, Rescale

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from run import split_train_val

from layers import transformation_from_parameters

from geometry import gramian, kern_mat, SSIM, reproject_image, fill_partial_depth
from geometry_plot import draw3DPts
from dataloader import pose_from_euler_t_Tensor, pose_RtT_from_se3_tensor

class Trainer:
    def __init__(self):
        ######################### model #############################
        self.num_pose_frames = 2
        self.num_layers = 18
        self.device = torch.device("cuda:1")
        # self.weights_init = "pretrained"
        self.weights_init = "not-pretrained"

        self.models = {}
        self.parameters_to_train =[]
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.num_layers,
            self.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)
        ### What about not training the pretrained part?
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=1 ) # originally 2 in monodepth2
        
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())


        ######################### optimizer #############################
        self.lr = 1e-4
        self.optim = torch.optim.Adam(self.parameters_to_train, lr=self.lr) #cefault 1e-3

        ########################## dataset ####################################
        ### for TUM dataset, you need to run associate.py on a folder of unzipped folders of TUM data sequences
        self.width = 128 # (72*96) (96, 128) (240, 320)
        self.height = 96
        source='TUM'
        if source=='CARLA':
            root_dir = root_dir = '/mnt/storage/minghanz_data/CARLA(with_pose)/_out'
        elif source == 'TUM':
            root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'

        self.kernalize = True
        self.models["loss"] = innerProdLoss(self.device, self.width, self.height, kernalize=self.kernalize, source=source)


        # from segmentation_models_pytorch.encoders import get_preprocessing_fn
        # preprocess_input_fn = get_preprocessing_fn('resnet18', pretrained='imagenet')
        preprocess_input_fn = None

        ### required as in https://pytorch.org/docs/stable/torchvision/models.html
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )

        img_pose_dataset = ImgPoseDataset(
            root_dir = root_dir, 
            transform=torchvision.transforms.Compose([Rescale(output_size=(self.height, self.width), post_fn=preprocess_input_fn), ToTensor(device=self.device) ]) )
        #### the image rgb is in the range of 0~1 for TUM dataset

        self.data_loader_train, self.data_loader_val = split_train_val(img_pose_dataset, 0.1)

        self.epochs = 1
        self.writer = SummaryWriter()

    def train(self):

        print('going into loop')
        iter_overall = 0
        lr_change_time = 0
        for i_epoch in range(self.epochs):
            print('entering epoch', i_epoch) 
            for i_batch, sample_batch in enumerate(self.data_loader_train):
                if i_batch > 10000:
                    break
                img1 = sample_batch['image 1']
                img2 = sample_batch['image 2']
                img1_raw = sample_batch['image 1 raw']
                img2_raw = sample_batch['image 2 raw']

                ### needed for using the pretrained model by torchvision
                for i in range(img1_raw.shape[0]):
                    img1[i] = self.normalize(img1_raw[i])
                    img2[i] = self.normalize(img2_raw[i])

                # print('img1 range: {:.2f}~{:.2f}'.format(img1_raw.min(), img1_raw.max() ) )
                # print('img2 range: {:.2f}~{:.2f}'.format(img2_raw.min(), img2_raw.max() ) )
                
                dep1 = sample_batch['depth 1']
                dep2 = sample_batch['depth 2']
                idep1 = sample_batch['idepth 1']
                idep2 = sample_batch['idepth 2']
                pose1_2 = sample_batch['rela_pose']
                euler1_2 = sample_batch['rela_euler']

                ################### network processing ################################
                ### Input to the network is preprocessed images
                ### The one used to warp is raw images
                pose_inputs = torch.cat([img1, img2], dim=1)
                pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                # axisangle, translation = self.models["pose"](pose_inputs)
                # print(axisangle.shape)
                # print(translation.shape)
                # T = transformation_from_parameters(axisangle[:,0], translation[:,0])
                axis_trans = self.models["pose"](pose_inputs)
                
                # axis_trans = axis_trans.squeeze(2)
                # axis_trans = axis_trans.squeeze(1)
                rot = axis_trans[...,:3]
                trans = axis_trans[...,3:]
                # trans_rot = torch.cat([trans, rot], dim=1)
                
                # R, t, T = pose_RtT_from_se3_tensor(rot, trans)
                # T = pose_from_euler_t_Tensor(axis_trans, device=self.device)

                T = transformation_from_parameters(
                        rot[:, 0], trans[:, 0] )
                T_gt = pose_from_euler_t_Tensor(euler1_2, device=self.device)

                ########################################################################

                #######################loss calculation ##############################

                # loss_dist, loss_dist_gt = self.models["loss"](dep1, dep2, T, img1_raw, img2_raw, T_gt)
                # loss_to_min = loss_dist - loss_dist_gt
                
                loss_to_min, loss_gt, output_warped = self.models["loss"](dep1, dep2, T, img1_raw, img2_raw, T_gt)

                #########################################################################
                ### record
                # self.writer.add_scalar('loss', loss_to_min, iter_overall)
                self.writer.add_scalars('pho_loss', {'pred': loss_to_min}, iter_overall)
                self.writer.add_scalars('pho_loss', {'gt': loss_gt}, iter_overall)

                # self.writer.add_scalars('cos_angle', {'pred': loss_dist}, iter_overall)
                # self.writer.add_scalars('cos_angle', {'gt': loss_dist_gt}, iter_overall)


                # image = vis.capture_screen_float_buffer()

                ### optimize
                self.optim.zero_grad()
                loss_to_min.backward()
                self.optim.step()

                if iter_overall % 50 == 0:
                    # print('Iteration', iter_overall, 'loss: {:.5f}'.format(loss_dist), 'loss_gt: {:5f}'.format(loss_dist_gt), 'diff: {:.5f}'.format(loss_to_min) )
                    print('Iteration', iter_overall, 'loss: {:.5f}'.format(loss_to_min), 'loss_gt: {:.5f}'.format(loss_gt)  )
                    # print('pred:\n', trans_rot)
                    # print('gt:\n', euler1_2)
                    
                    warped_img1 = output_warped[0]['reconstructed']
                    warped_img2 = output_warped[1]['reconstructed']
                    warped_img1_gt = output_warped[0]['reconstructed_gt']
                    warped_img2_gt = output_warped[1]['reconstructed_gt']

                    grid1 = torchvision.utils.make_grid(img1_raw)
                    grid2 = torchvision.utils.make_grid(img2_raw)
                    self.writer.add_image('img1',grid1, iter_overall)
                    self.writer.add_image('img2',grid2, iter_overall)

                    grid1 = torchvision.utils.make_grid(warped_img1)
                    grid2 = torchvision.utils.make_grid(warped_img2)
                    self.writer.add_image('warped img1',grid1, iter_overall)
                    self.writer.add_image('warped img2',grid2, iter_overall)

                    grid1 = torchvision.utils.make_grid(warped_img1_gt)
                    grid2 = torchvision.utils.make_grid(warped_img2_gt)
                    self.writer.add_image('warped img1 gt',grid1, iter_overall)
                    self.writer.add_image('warped img2 gt',grid2, iter_overall)

                iter_overall += 1


                
class innerProdLoss(nn.Module):
    def __init__(self, device, width=96, height=72, kernalize=False, source='CARLA'):
        super(innerProdLoss, self).__init__()
        self.device = device
        # K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
        # K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
        # input: x front, y right (aligned with u direction), z down (aligned with v direction) [CARLA definition]
        # output: u, v, 1
        self.source = source
        self.options_from_source(width=width, height=height)

        inv_K = torch.Tensor(np.linalg.inv(self.K)).to(self.device)
        self.K = torch.Tensor(self.K).to(self.device)

        u_grid = torch.Tensor(np.arange(width) )
        v_grid = torch.Tensor(np.arange(height) )
        uu_grid = u_grid.unsqueeze(0).expand((height, -1) ).reshape(-1)
        vv_grid = v_grid.unsqueeze(1).expand((-1, width) ).reshape(-1)
        self.width = width
        self.height = height
        self.uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N, correspond to u,v,1
        self.yz1_grid = torch.mm(inv_K, self.uv1_grid).to(self.device) # 3*N, corresponding to x,y,z
        self.kernalize = kernalize
        self.dist_coef_color = 1e-1

        self.ssim = SSIM()

    def options_from_source(self, width, height):
        '''
        CARLA and TUM have differenct definition of relation between xyz coordinate and uv coordinate.
        CARLA xyz is front-right(u)-down(v)(originally up, which is left handed, fixed to down in pose_from_euler_t function)
        TUM xyz is right(u)-down(v)-front
        dist_coef is scaling in the exponential in the RBF kernel, related to the movement and scenario scale in the data
        '''
        assert self.source == 'CARLA' or self.source == 'TUM', 'source unrecognized'
        if self.source == 'CARLA':
            fx=int(width/2)
            fy=int(width/2)
            cx=int(width/2)
            cy=int(height/2)
            self.K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
            self.dist_coef = 1e-1
        elif self.source == 'TUM':
            fx = width/640.0*525.0  # focal length x
            fy = height/480.0*525.0  # focal length y
            cx = width/640.0*319.5  # optical center x
            cy = height/480.0*239.5  # optical center y
            self.K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
            self.dist_coef = 5e-2
        
    def forward(self, depth1, depth2, pose1_2, img1, img2, pose1_2_gt=None): # img1 and img2 only for visualization
        
        ### flatten depth and feature maps ahead of other operations to make pixel selection easier
        ### add a pixel selection module for the cases where not all pixels are included in the CVO loss
        batch_size = depth1.shape[0]
        n_pts_1 = depth1.shape[2]*depth1.shape[3]
        n_pts_2 = depth2.shape[2]*depth2.shape[3]

        depth1_flat = depth1.reshape(-1, 1, n_pts_1) # B*1*N  # .expand(-1, 3, -1) # B*3*N
        depth2_flat = depth2.reshape(-1, 1, n_pts_2) # B*1*N  # .expand(-1, 3, -1)

        img_flat_1 = img1.reshape(batch_size, 3, n_pts_1) # B*C*N1
        img_flat_2 = img2.reshape(batch_size, 3, n_pts_2) # B*C*N2

        inout_flat = {}
        inout_flat[0] = {}
        inout_flat[1] = {}
        inout_flat[0]['depth'] = depth1_flat
        inout_flat[1]['depth'] = depth2_flat
        inout_flat[0]['img'] = img_flat_1
        inout_flat[1]['img'] = img_flat_2

        pose_list = [pose1_2.inverse(), pose1_2]
        pose_gt_list = [pose1_2_gt.inverse(), pose1_2_gt]

        xyz_flat = self.xyz_trans(inout_flat, pose_list, pose_gt_list)

        ###### photometric error
        photometric_loss, pho_loss_gt, output_warped = self.pho_loss(inout_flat, xyz_flat)

        return photometric_loss, pho_loss_gt, output_warped

        # #################################################
        # inner_neg = torch.tensor(0., device=self.device)
        # inner_neg_gt = torch.tensor(0., device=self.device)        

        # for i in range(batch_size):
        #     xyz1, dep_flat_1_sel, img_flat_1_sel = self.selected(depth1_flat[i], img_flat_1[i]) #3*N
        #     xyz2, dep_flat_2_sel, img_flat_2_sel = self.selected(depth2_flat[i], img_flat_2[i]) #3*N 

        #     inner_neg_single, inner_neg_single_gt = self.calc_loss(xyz1, xyz2, pose1_2[i], img_flat_1_sel, img_flat_2_sel, pose1_2_gt[i])

        #     inner_neg = inner_neg + inner_neg_single
        #     inner_neg_gt = inner_neg_gt + inner_neg_single_gt
        
        # return inner_neg, inner_neg_gt

    def xyz_trans(self, inout_flat, pose_list, pose_gt_list):

        xyz_flat = {}
        xyz_flat[0] = {}
        xyz_flat[1] = {}
        
        ## generate xyz for all pixels B*3*N
        for i in range(2):
            ## fill invalid depth with a dummy value, but preserve a mask recording which pixels are valid
            xyz_flat[i]['mask_valid_depth'] = inout_flat[i]['depth'] > 0 # B*1*N
            xyz_flat[i]['depth_filled'] = fill_partial_depth( inout_flat[i]['depth'] )
            xyz_flat[i]['xyz'] = self.yz1_grid * xyz_flat[i]['depth_filled'] 
            
            xyz_homo = torch.cat( ( xyz_flat[i]['xyz'], torch.ones((xyz_flat[i]['xyz'].shape[0], 1, xyz_flat[i]['xyz'].shape[2])).to(self.device) ), dim=1) # B*4*N
            xyz_flat[i]['xyz_trans_homo'] = torch.matmul(pose_list[i], xyz_homo) # B*4*N
            xyz_flat[i]['xyz_trans'] = xyz_flat[i]['xyz_trans_homo'][:, 0:3, :] # B*3*N

            xyz_flat[i]['xyz_trans_homo_gt'] = torch.matmul(pose_gt_list[i], xyz_homo) # B*4*N
            xyz_flat[i]['xyz_trans_gt']  = xyz_flat[i]['xyz_trans_homo_gt'][:, 0:3, :] # B*3*N

        return xyz_flat

    def pho_loss(self, inout_flat, xyz_flat):

        xyz1_trans = xyz_flat[0]['xyz_trans'] #B*C*N
        xyz2_trans = xyz_flat[1]['xyz_trans'] #B*C*N
        img1 = inout_flat[0]['img'] #B*1*N
        img2 = inout_flat[1]['img']
        img1 = img1.reshape(img1.shape[0], img1.shape[1], self.height, self.width) # B*C*H*W
        img2 = img2.reshape(img2.shape[0], img2.shape[1], self.height, self.width)
        
        ### reprojection
        reconstructed_img1 = reproject_image(xyz1_trans, self.K, img2, zdim=2) # B*C*H*W
        reconstructed_img2 = reproject_image(xyz2_trans, self.K, img1, zdim=2) # B*C*H*W

        ### calculate loss B*1*H*W
        reproj_diff_1 = self.compute_reprojection_loss(reconstructed_img1, img1)
        reproj_diff_2 = self.compute_reprojection_loss(reconstructed_img2, img2)

        ### exclude the pixels with invalid depth value
        mask_1_pos = xyz_flat[0]['mask_valid_depth'] # B*1*N
        mask_2_pos = xyz_flat[1]['mask_valid_depth']
        valid_num_1 = mask_1_pos.sum()
        valid_num_2 = mask_2_pos.sum()
        mask_1 = mask_1_pos == False
        mask_2 = mask_2_pos == False
        mask_1 = mask_1.reshape(mask_1.shape[0], mask_1.shape[1], self.height, self.width) # B*1*H*W
        mask_2 = mask_2.reshape(mask_2.shape[0], mask_2.shape[1], self.height, self.width)

        reproj_diff_1[mask_1] = 0
        reproj_diff_2[mask_2] = 0
        for i in range(reconstructed_img1.shape[1]):
            reconstructed_img1[:,i:i+1,:,:][mask_1] = 0
            reconstructed_img2[:,i:i+1,:,:][mask_2] = 0

        loss = torch.sum(reproj_diff_1)/valid_num_1 + torch.sum(reproj_diff_2)/valid_num_2


        xyz1_trans_gt = xyz_flat[0]['xyz_trans_gt'] #B*C*N
        xyz2_trans_gt = xyz_flat[1]['xyz_trans_gt'] #B*C*N
        
        reconstructed_img1_gt = reproject_image(xyz1_trans_gt, self.K, img2, zdim=2) # B*C*H*W
        reconstructed_img2_gt = reproject_image(xyz2_trans_gt, self.K, img1, zdim=2) # B*C*H*W

        reproj_diff_1_gt = self.compute_reprojection_loss(reconstructed_img1_gt, img1)
        reproj_diff_2_gt = self.compute_reprojection_loss(reconstructed_img2_gt, img2)

        reproj_diff_1_gt[mask_1] = 0
        reproj_diff_2_gt[mask_2] = 0
        for i in range(reconstructed_img1.shape[1]):
            reconstructed_img1_gt[:,i:i+1,:,:][mask_1] = 0
            reconstructed_img2_gt[:,i:i+1,:,:][mask_2] = 0

        loss_gt = torch.sum(reproj_diff_1_gt)/valid_num_1 + torch.sum(reproj_diff_2_gt)/valid_num_2

        output = {}
        output[0] = {}
        output[1] = {}

        output[0]['reconstructed'] = reconstructed_img1
        output[1]['reconstructed'] = reconstructed_img2
        output[0]['reconstructed_gt'] = reconstructed_img1_gt
        output[1]['reconstructed_gt'] = reconstructed_img2_gt
        

        return loss, loss_gt, output

    def compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        Return: B*1*H*W
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        reprojection_loss = l1_loss

        # if self.opt.no_ssim:
            # reprojection_loss = l1_loss
        # else:
        #     ssim_loss = self.ssim(pred, target).mean(1, True)
        #     reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss     

    def calc_loss(self, xyz1, xyz2, pose1_2, img1, img2, pose1_2_gt=None):



        ### color inner product
        gramian_color, _ = gramian(img1, img2, norm_mode=False, kernalize=self.kernalize, norm_dim=0, dist_coef=self.dist_coef_color )
        gramian_c1, _ = gramian(img1, img1, norm_mode=False, kernalize=self.kernalize, norm_dim=0, dist_coef=self.dist_coef_color )
        gramian_c2, _ = gramian(img2, img2, norm_mode=False, kernalize=self.kernalize, norm_dim=0, dist_coef=self.dist_coef_color )

        ### xyz inner product
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)
        pcl_diff_1 = kern_mat(xyz1, xyz1, self.dist_coef)
        pcl_diff_2 = kern_mat(xyz2_trans, xyz2_trans, self.dist_coef)

        f1_f1 = torch.sum(pcl_diff_1 * gramian_c1 ) 
        f2_f2 = torch.sum(pcl_diff_2 * gramian_c2 )
        f1_f2 = torch.sum(pcl_diff_exp * gramian_color )
        inner_neg = f1_f1 + f2_f2 - 2*f1_f2
        cos_angle = f1_f2 / torch.sqrt(f1_f1 * f2_f2 )

        pcl_diff_exp_gt = kern_mat(xyz1, xyz2_trans_gt, self.dist_coef)
        pcl_diff_2_gt = kern_mat(xyz2_trans_gt, xyz2_trans_gt, self.dist_coef)

        f2_f2_gt = torch.sum(pcl_diff_2_gt * gramian_c2 )
        f1_f2_gt = torch.sum(pcl_diff_exp_gt * gramian_color )
        # inner_neg_gt = f1_f1 + f2_f2_gt - 2*f1_f2_gt
        cos_angle_gt = f1_f2_gt / torch.sqrt(f1_f1 * f2_f2_gt )

        # loss = inner_neg / inner_neg_gt

        loss = -cos_angle
        loss_gt = -cos_angle_gt
        # print('true:', float(cos_angle_gt), 'pred:', float(cos_angle) )

        # inner_neg = torch.sum(pcl_diff_1 * gramian_c1 ) + torch.sum(pcl_diff_2 * gramian_c2 ) - 2 * torch.sum(pcl_diff_exp * gramian_color )

        ## visualization
        # print('func_dist:', inner_neg)
        # draw3DPts(xyz1.detach(), xyz2_trans.detach(), img1.detach(), img2.detach())
        # print('func_dist_gt:', inner_neg_gt)
        # draw3DPts(xyz1.detach(), xyz2_trans_gt.detach(), img1.detach(), img2.detach())

        return loss, loss_gt



    def selected(self, depth_flat_sample, img_flat_sample):
        '''
        Input is 1*N or 3*N
        Output 1*C*N (keep the batch dimension)
        '''
        # mask = (depth_flat_sample.squeeze() > 0) & (self.uv1_grid[0] > 4) & (self.uv1_grid[0] < self.width - 5) & (self.uv1_grid[1] > 4) & (self.uv1_grid[1] < self.height - 5)
        mask = (depth_flat_sample.squeeze() > 0)

        depth_flat_sample_selected = depth_flat_sample[:,mask]
        img_flat_sample_selected = img_flat_sample[:,mask]

        yz1_grid_selected = self.yz1_grid[:,mask]
        xyz_selected = yz1_grid_selected * depth_flat_sample_selected

        return xyz_selected.unsqueeze(0), depth_flat_sample_selected.unsqueeze(0), img_flat_sample_selected.unsqueeze(0)

if __name__ == "__main__":
    pose_predictor = Trainer()
    pose_predictor.train()