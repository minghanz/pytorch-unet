import torch
import torch.nn as nn
from unet import UNet
from unet_pose_regressor import UNetRegressor
import numpy as np
from dataloader import pose_from_euler_t, pose_from_euler_t_Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from geometry_plot import draw3DPts

from geometry import kern_mat, gramian, gen_3D, gen_3D_flat

class UNetInnerProd(nn.Module):
    def __init__(self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        device= torch.device('cuda'),
        width=96, 
        height=72, 
        diff_mode=True,
        kernalize=False,
        sparsify=False,
        color_in_cost=False, 
        L2_norm=True, 
        pose_predict_mode=False, 
        source='TUM'
    ):
        super(UNetInnerProd, self).__init__()
        self.device = device
        self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)
        self.model_loss = innerProdLoss(device, width, height, diff_mode, kernalize, sparsify, 
            color_in_cost, L2_norm, source).to(device)
        self.pose_predict_mode = pose_predict_mode
        if self.pose_predict_mode:
            self.pose_predictor = UNetRegressor(batch_norm=True, feature_channels=n_classes*2).to(device)

    def forward(self, img1, img2, dep1, dep2, idep1, idep2, pose1_2):
        feature1 = self.model_UNet(img1)
        feature2 = self.model_UNet(img2)

        dep1.requires_grad = False
        dep2.requires_grad = False
        pose1_2.requires_grad = False

        if self.pose_predict_mode:
            # with pose predictor
            euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
            pose1_2_pred = pose_from_euler_t_Tensor(euler_pred, device=self.device)

            loss, innerp_loss, feat_norm, innerp_loss_pred = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2, pose1_2_pred)
            return feature1, feature2, loss, innerp_loss, feat_norm, innerp_loss_pred, euler_pred
        else:
            # without pose predictor
            loss, innerp_loss, feat_norm = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2)
            return feature1, feature2, loss, innerp_loss, feat_norm

    def set_norm_level(self, i_batch):
        self.model_loss.set_norm_level(i_batch)
        return


class innerProdLoss(nn.Module):
    def __init__(self, device, width=96, height=72, diff_mode=True, kernalize=False, sparsify=False,
            color_in_cost=False, L2_norm=True, source='CARLA'):
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
        uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N, correspond to u,v,1
        self.yz1_grid = torch.mm(inv_K, uv1_grid).to(self.device) # 3*N, corresponding to x,y,z
        self.diff_mode = diff_mode
        self.kernalize = kernalize
        self.color_in_cost = color_in_cost
        self.L2_norm = L2_norm
        self.sparsify = sparsify

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
            self.noise_trans_scale = 1.0
        elif self.source == 'TUM':
            fx = width/640.0*525.0  # focal length x
            fy = height/480.0*525.0  # focal length y
            cx = width/640.0*319.5  # optical center x
            cy = height/480.0*239.5  # optical center y
            self.K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
            self.dist_coef = 1e-0
            self.noise_trans_scale = 0.1

    def gen_rand_pose(self):
        # trans_noise = np.random.normal(scale=self.noise_trans_scale, size=(3,))
        # rot_noise = np.random.normal(scale=3.0, size=(3,))
        trans_noise = torch.randn((3), dtype=torch.float, device=self.device) * self.noise_trans_scale
        rot_noise = torch.randn((3), dtype=torch.float, device=self.device) * 3.0
        
        if self.source=='CARLA':
            # self.pose1_2_noise = pose_from_euler_t(trans_noise[0], 3*trans_noise[1], trans_noise[2], 0, rot_noise[1], 3*rot_noise[2]) # for carla
            noise_euler_tensor = torch.stack([ trans_noise[0], 3*trans_noise[1], trans_noise[2], 0, rot_noise[1], 3*rot_noise[2] ]).unsqueeze(0)
            self.pose1_2_noise = pose_from_euler_t_Tensor(noise_euler_tensor, device=self.device).squeeze(0)
        elif self.source=='TUM':
            # self.pose1_2_noise = pose_from_euler_t(trans_noise[0], trans_noise[1], trans_noise[2], rot_noise[0], rot_noise[1], rot_noise[2]) # for TUM
            noise_euler_tensor = torch.stack([trans_noise[0], trans_noise[1], trans_noise[2], rot_noise[0], rot_noise[1], rot_noise[2]]).unsqueeze(0)
            self.pose1_2_noise = pose_from_euler_t_Tensor(noise_euler_tensor, device=self.device).squeeze(0)

        # self.pose1_2_noise = torch.tensor(self.pose1_2_noise).to(self.device)
        # self.pose1_2_noise = torch.tensor(self.pose1_2_noise, dtype=torch.float, device=self.device)
        self.pose1_2_noise.requires_grad = False

    def set_norm_level(self, i_batch):
        if i_batch < 1000:
            self.norm_scale = 0
        elif i_batch < 2000:
            self.norm_scale = 1e-3
        elif i_batch < 3000:
            self.norm_scale = 1e-2
        # elif i_batch < 2000:
        #     self.norm_scale = 1
        else:
            self.norm_scale = 1e-1

        # if self.sparsify:
        #     self.norm_scale *= 1e-2

    def calc_loss(self, xyz1, xyz2, pose1_2, feature1, feature2, img1, img2, pose1_2_pred=None):
        '''
        Input points are flat
        '''
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
        xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N

        if self.sparsify:
            norm_mode = 0
        else:
            norm_mode = 1
        gramian_feat, fea_norm_sum = gramian(feature1, feature2, norm_mode=norm_mode, kernalize=self.kernalize, L2_norm=self.L2_norm) # could be kernelized or not
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)

        if self.color_in_cost:
            gramian_color, _ = gramian(img1, img2, norm_mode=0, kernalize=self.kernalize, L2_norm=False)
        
        # print('now original')
        # draw3DPts(xyz1, xyz2, img1, img2)
        # print('now matched')
        # draw3DPts(xyz1, xyz2_trans, img1, img2)

        if self.sparsify:
            fea_norm_sum = fea_norm_sum * self.norm_scale

        if self.diff_mode:
            if self.source=='TUM':
                ### 1. perturb the scenario points by moving them to the center first
                mean_trans_2 = xyz2_trans.mean(dim=(0,2))
                # trans_back = pose_from_euler_t(-mean_trans_2[0], -mean_trans_2[1], -mean_trans_2[2], 0, 0, 0)
                # # trans_back = torch.Tensor(trans_back).to(self.device)
                # trans_back = torch.tensor(trans_back, dtype=torch.float, device=self.device)
                trans = torch.tensor([0., 0., 0.], dtype=torch.float, device=self.device)

                trans_rot = torch.stack([-mean_trans_2[0], -mean_trans_2[1], -mean_trans_2[2]])
                trans_euler = torch.cat((trans_rot, trans), dim=0 ).unsqueeze(0)
                trans_back = pose_from_euler_t_Tensor(trans_euler, device=self.device).squeeze(0)

                trans_back.requires_grad = False

                # trans_forth = pose_from_euler_t(mean_trans_2[0], mean_trans_2[1], mean_trans_2[2], 0, 0, 0)
                # # trans_forth = torch.Tensor(trans_forth).to(self.device)
                # trans_forth = torch.tensor(trans_forth, dtype=torch.float, device=self.device)

                trans_rot = torch.stack([mean_trans_2[0], mean_trans_2[1], mean_trans_2[2]])
                trans_euler = torch.cat((trans_rot, trans), dim=0 ).unsqueeze(0)
                trans_forth = pose_from_euler_t_Tensor(trans_euler, device=self.device).squeeze(0)

                trans_forth.requires_grad = False

                pose1_2_noisy = torch.matmul(trans_forth, torch.matmul(self.pose1_2_noise, trans_back) )
                xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_trans_homo)[:, 0:3, :] # B*3*N

            elif self.source=='CARLA':
                ## 2. directly perturb camera pose
                pose1_2_noisy = torch.matmul(pose1_2, self.pose1_2_noise)
                xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_homo)[:, 0:3, :] # B*3*N

            pcl_diff_exp_noisy = kern_mat(xyz1, xyz2_trans_noisy, self.dist_coef)

            ### 3. treat original points (without matching) as perturbation
            # pcl_diff_exp_noisy = kern_mat(xyz1, xyz2, self.dist_coef)

            # pcl_diff_exp_diff = pcl_diff_exp - pcl_diff_exp_noisy
            if self.color_in_cost:
                # inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat * gramian_color ) #  * gramian_feat
                inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat * gramian_color )
                inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat * gramian_color )
                inner_neg = inner_neg_noisy / inner_neg_match
            else:
                # inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat ) #  * gramian_feat
                inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat  )
                inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat )
                inner_neg = inner_neg_noisy / inner_neg_match

            final_loss = inner_neg
            if self.sparsify:
                final_loss = final_loss + fea_norm_sum

            # print('inner_neg', inner_neg)
            # print('now noisy')
            # draw3DPts(xyz1, xyz2_trans_noisy, img1, img2)

        else:
            if self.color_in_cost:
                inner_neg = - torch.mean(pcl_diff_exp * gramian_feat * gramian_color ) # * gramian_feat
            else:
                inner_neg = - torch.mean(pcl_diff_exp * gramian_feat ) # * gramian_feat

            final_loss = inner_neg
            if self.sparsify:
                final_loss = final_loss + fea_norm_sum
        
        if pose1_2_pred is not None:
            xyz2_trans_pred = torch.matmul(pose1_2_pred, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred, self.dist_coef)
            if self.color_in_cost:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat * gramian_color )
            else:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat ) 

            final_loss = final_loss + inner_neg_pred

            
            return final_loss, inner_neg, fea_norm_sum, inner_neg_pred

        return final_loss, inner_neg, fea_norm_sum


    def forward(self, feature1, feature2, depth1, depth2, pose1_2, img1, img2, pose1_2_pred=None): # img1 and img2 only for visualization
        
        ### flatten depth and feature maps ahead of other operations to make pixel selection easier
        ### add a pixel selection module for the cases where not all pixels are included in the CVO loss
        batch_size = depth1.shape[0]
        n_pts_1 = depth1.shape[2]*depth1.shape[3]
        n_pts_2 = depth2.shape[2]*depth2.shape[3]

        depth1_flat = depth1.reshape(-1, 1, n_pts_1) # B*1*N  # .expand(-1, 3, -1) # B*3*N
        depth2_flat = depth2.reshape(-1, 1, n_pts_2) # B*1*N  # .expand(-1, 3, -1)

        channels = feature1.shape[1]

        fea_flat_1 = feature1.reshape(batch_size, channels, n_pts_1) # B*C*N1
        fea_flat_2 = feature2.reshape(batch_size, channels, n_pts_2) # B*C*N2

        img_flat_1 = img1.reshape(batch_size, 3, n_pts_1) # B*C*N1
        img_flat_2 = img2.reshape(batch_size, 3, n_pts_2) # B*C*N2

        #################################################
        final_loss = torch.tensor(0., device=self.device)
        inner_neg = torch.tensor(0., device=self.device)
        fea_norm_sum = torch.tensor(0., device=self.device)
        inner_neg_pred = torch.tensor(0., device=self.device)

        for i in range(batch_size):
            xyz1, dep_flat_1_sel, fea_flat_1_sel, img_flat_1_sel = selected(depth1_flat[i], fea_flat_1[i], img_flat_1[i], self.yz1_grid) #3*N
            xyz2, dep_flat_2_sel, fea_flat_2_sel, img_flat_2_sel = selected(depth2_flat[i], fea_flat_2[i], img_flat_2[i], self.yz1_grid) #3*N            

            if pose1_2_pred is None:
                final_loss_single, inner_neg_single, fea_norm_sum_single = \
                    self.calc_loss(xyz1, xyz2, pose1_2[i], fea_flat_1_sel, fea_flat_2_sel, img_flat_1_sel, img_flat_2_sel)
            else:
                final_loss_single, inner_neg_single, fea_norm_sum_single, inner_neg_pred_single = \
                    self.calc_loss(xyz1, xyz2, pose1_2[i], fea_flat_1_sel, fea_flat_2_sel, img_flat_1_sel, img_flat_2_sel, pose1_2_pred[i])
                inner_neg_pred = inner_neg_pred + inner_neg_pred_single

            final_loss = final_loss + final_loss_single
            inner_neg = inner_neg + inner_neg_single
            fea_norm_sum = fea_norm_sum + fea_norm_sum_single
        
        if pose1_2_pred is None:
            return final_loss, inner_neg, fea_norm_sum
        else:
            return final_loss, inner_neg, fea_norm_sum, inner_neg_pred

            
        #############################################
        # xyz1, xyz2 = gen_3D_flat(self.yz1_grid, depth1_flat, depth2_flat)

        # return self.calc_loss(xyz1, xyz2, pose1_2, fea_flat_1, fea_flat_2, img_flat_1, img_flat_2, pose1_2_pred)


def selected(depth_flat_sample, fea_flat_sample, img_flat_sample, yz1_grid):
    '''
    Input is 1*N or 3*N
    Output 1*C*N (keep the batch dimension)
    '''
    mask = depth_flat_sample.squeeze() > 0

    depth_flat_sample_selected = depth_flat_sample[:,mask]
    fea_flat_sample_selected = fea_flat_sample[:,mask]
    img_flat_sample_selected = img_flat_sample[:,mask]

    yz1_grid_selected = yz1_grid[:,mask]
    xyz_selected = yz1_grid_selected * depth_flat_sample_selected

    return xyz_selected.unsqueeze(0), depth_flat_sample_selected.unsqueeze(0), fea_flat_sample_selected.unsqueeze(0), img_flat_sample_selected.unsqueeze(0)