
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networks

from dataloader import ImgPoseDataset, ToTensor, Rescale

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from run import split_train_val

from layers import transformation_from_parameters

from geometry import gramian, kern_mat
from geometry_plot import draw3DPts
from dataloader import pose_from_euler_t_Tensor

class Trainer:
    def __init__(self):
        ######################### model #############################
        self.num_pose_frames = 2
        self.num_layers = 18
        self.device = torch.device("cuda:1")
        self.weights_init = "pretrained"

        self.models = {}
        self.parameters_to_train =[]
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.num_layers,
            self.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)
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
        # preprocess_input_fn = get_preprocessing_fn('resnet34', pretrained='imagenet')
        preprocess_input_fn = None

        img_pose_dataset = ImgPoseDataset(
            root_dir = root_dir, 
            transform=torchvision.transforms.Compose([Rescale(output_size=(self.height, self.width), post_fn=preprocess_input_fn), ToTensor(device=self.device) ]) )
        
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
                
                dep1 = sample_batch['depth 1']
                dep2 = sample_batch['depth 2']
                idep1 = sample_batch['idepth 1']
                idep2 = sample_batch['idepth 2']
                pose1_2 = sample_batch['rela_pose']
                euler1_2 = sample_batch['rela_euler']

                ### network processing
                pose_inputs = torch.cat([img1_raw, img2_raw], dim=1)
                pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                # axisangle, translation = self.models["pose"](pose_inputs)
                # print(axisangle.shape)
                # print(translation.shape)
                # T = transformation_from_parameters(axisangle[:,0], translation[:,0])
                axis_trans = self.models["pose"](pose_inputs)
                axis_trans = axis_trans.squeeze(2)
                axis_trans = axis_trans.squeeze(1)
                rot = axis_trans[...,:3]
                trans = axis_trans[...,3:]
                trans_rot = torch.cat([trans, rot], dim=1)
                
                T = pose_from_euler_t_Tensor(trans_rot, device=self.device)
                T_gt = pose_from_euler_t_Tensor(euler1_2, device=self.device)
                loss_dist = self.models["loss"](dep1, dep2, T, img1_raw, img2_raw, T_gt)

                ### record
                self.writer.add_scalars('loss', {'pred': loss_dist}, iter_overall)


                # image = vis.capture_screen_float_buffer()

                ### optimize
                self.optim.zero_grad()
                loss_dist.backward()
                self.optim.step()

                if iter_overall % 50 == 0:
                    print('Iteration', iter_overall, 'loss:', loss_dist)
                    print('pred:\n', trans_rot)
                    print('gt:\n', euler1_2)
                    
                    grid1 = torchvision.utils.make_grid(img1_raw)
                    grid2 = torchvision.utils.make_grid(img2_raw)
                    self.writer.add_image('img1',grid1, iter_overall)
                    self.writer.add_image('img2',grid2, iter_overall)

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
            self.dist_coef = 3e-1
        
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

        #################################################
        inner_neg = torch.tensor(0., device=self.device)

        for i in range(batch_size):
            xyz1, dep_flat_1_sel, img_flat_1_sel = self.selected(depth1_flat[i], img_flat_1[i]) #3*N
            xyz2, dep_flat_2_sel, img_flat_2_sel = self.selected(depth2_flat[i], img_flat_2[i]) #3*N 

            inner_neg_single = self.calc_loss(xyz1, xyz2, pose1_2[i], img_flat_1_sel, img_flat_2_sel, pose1_2_gt[i])

            inner_neg = inner_neg + inner_neg_single
        
        return inner_neg

    def calc_loss(self, xyz1, xyz2, pose1_2, img1, img2, pose1_2_gt=None):
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
        xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N

        xyz2_trans_homo_gt = torch.matmul(pose1_2_gt, xyz2_homo) # B*4*N
        xyz2_trans_gt = xyz2_trans_homo_gt[:, 0:3, :] # B*3*N

        ### color inner product
        gramian_color, _ = gramian(img1, img2, norm_mode=False, kernalize=self.kernalize, norm_dim=0 )
        gramian_c1, _ = gramian(img1, img1, norm_mode=False, kernalize=self.kernalize, norm_dim=0 )
        gramian_c2, _ = gramian(img2, img2, norm_mode=False, kernalize=self.kernalize, norm_dim=0 )

        ### xyz inner product
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)
        pcl_diff_1 = kern_mat(xyz1, xyz1, self.dist_coef)
        pcl_diff_2 = kern_mat(xyz2_trans, xyz2_trans, self.dist_coef)

        pcl_diff_exp_gt = kern_mat(xyz1, xyz2_trans_gt, self.dist_coef)
        pcl_diff_2_gt = kern_mat(xyz2_trans_gt, xyz2_trans_gt, self.dist_coef)

        f1_f1 = torch.sum(pcl_diff_1 * gramian_c1 ) 
        f2_f2 = torch.sum(pcl_diff_2 * gramian_c2 )
        f1_f2 = torch.sum(pcl_diff_exp * gramian_color )
        inner_neg = f1_f1 + f2_f2 - 2*f1_f2

        f2_f2_gt = torch.sum(pcl_diff_2_gt * gramian_c2 )
        f1_f2_gt = torch.sum(pcl_diff_exp_gt * gramian_color )
        inner_neg_gt = f1_f1 + f2_f2_gt - 2*f1_f2_gt

        loss = inner_neg / inner_neg_gt

        # inner_neg = torch.sum(pcl_diff_1 * gramian_c1 ) + torch.sum(pcl_diff_2 * gramian_c2 ) - 2 * torch.sum(pcl_diff_exp * gramian_color )

        ### visualization
        # print('func_dist:', inner_neg)
        # draw3DPts(xyz1.detach(), xyz2_trans.detach(), img1.detach(), img2.detach())

        return loss



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