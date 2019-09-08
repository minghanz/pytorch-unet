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
        K=np.zeros((3,3)), 
        width=96, 
        height=72, 
        diff_mode=True,
        sparse_mode=False,
        kernalize=False,
        color_in_cost=False, 
        L2_norm=True, 
        pose_predict_mode=False, 
        dist_coef=1e-1
    ):
        super(UNetInnerProd, self).__init__()
        self.device = device
        self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)
        self.model_loss = innerProdLoss(device, K, width, height, diff_mode, sparse_mode, kernalize, color_in_cost, L2_norm, dist_coef).to(device)
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



class innerProdLoss(nn.Module):
    def __init__(self, device, K=np.zeros((3,3)), width=96, height=72, diff_mode=True, sparse_mode=False, kernalize=False, color_in_cost=False, L2_norm=True, dist_coef=1e-1):
        super(innerProdLoss, self).__init__()
        self.device = device
        # K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
        # K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
        # input: x front, y right (aligned with u direction), z down (aligned with v direction) [CARLA definition]
        # output: u, v, 1
        
        inv_K = torch.Tensor(np.linalg.inv(K)).to(self.device)
        K = torch.Tensor(K).to(self.device)

        u_grid = torch.Tensor(np.arange(width) )
        v_grid = torch.Tensor(np.arange(height) )
        uu_grid = u_grid.unsqueeze(0).expand((height, -1) ).reshape(-1)
        vv_grid = v_grid.unsqueeze(1).expand((-1, width) ).reshape(-1)
        uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N, correspond to u,v,1
        self.yz1_grid = torch.mm(inv_K, uv1_grid).to(self.device) # 3*N, corresponding to x,y,z
        self.diff_mode = diff_mode
        self.sparse_mode = sparse_mode
        self.kernalize = kernalize
        self.color_in_cost = color_in_cost
        self.L2_norm = L2_norm
        self.dist_coef = dist_coef

    def gen_rand_pose(self):
        trans_noise = np.random.normal(scale=1.0, size=(3,))
        rot_noise = np.random.normal(scale=5.0, size=(3,))
        self.pose1_2_noise = pose_from_euler_t(trans_noise[0], 3*trans_noise[1], trans_noise[2], 0, rot_noise[1], 3*rot_noise[2])
        self.pose1_2_noise = torch.Tensor(self.pose1_2_noise).to(self.device)
        self.pose1_2_noise.requires_grad = False

    def calc_loss(self, xyz1, xyz2, pose1_2, feature1, feature2, img1, img2, pose1_2_pred=None):
        '''
        Input points are flat
        '''
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans = torch.matmul(pose1_2, xyz2_homo)[:, 0:3, :] # B*3*N

        if self.sparse_mode:
            normalization_mode = 0
        else:
            normalization_mode = 1
        gramian_feat, fea_norm_sum = gramian(feature1, feature2, normalization_mode, kernalize=self.kernalize, L2_norm=self.L2_norm) # could be kernelized or not
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)

        if self.color_in_cost:
            gramian_color, _ = gramian(img1, img2, norm_mode=0, kernalize=self.kernalize, L2_norm=False)

        # n_pts_1 = img1.shape[2]*img1.shape[3]
        # n_pts_2 = img2.shape[2]*img2.shape[3]

        # img1_flat = img1.reshape(-1, 3, n_pts_1)
        # img2_flat = img2.reshape(-1, 3, n_pts_2)
        
        # print('now original')
        # draw3DPts(xyz1, xyz2, img1, img2)
        # print('now matched')
        # draw3DPts(xyz1, xyz2_trans, img1, img2)
        if self.L2_norm:
            if self.diff_mode:
                fea_norm_sum = fea_norm_sum *1e2
            else: 
                fea_norm_sum = fea_norm_sum *1e3
        else:
            if self.diff_mode:
                fea_norm_sum = fea_norm_sum *1e1
            else: 
                fea_norm_sum = fea_norm_sum *1e2

        if self.diff_mode:
            # pose1_2_noisy = torch.matmul(pose1_2, self.pose1_2_noise)
            # xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_homo)[:, 0:3, :] # B*3*N
            # pcl_diff_exp_noisy = kern_mat(xyz1, xyz2_trans_noisy, self.dist_coef)

            pcl_diff_exp_noisy = kern_mat(xyz1, xyz2, self.dist_coef)

            pcl_diff_exp_diff = pcl_diff_exp - pcl_diff_exp_noisy

            if self.color_in_cost:
                inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat * gramian_color )
            else:
                inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat )

            final_loss = inner_neg
            if self.sparse_mode:
                final_loss = inner_neg + fea_norm_sum

            # print('now noisy')
            # # draw3DPts(xyz1, xyz2_trans_noisy, img1, img2)
            # draw3DPts(xyz1, xyz2, img1, img2)

            # inner_neg = - torch.sum(pcl_diff_exp * gramian_feat ) #, dim=(1,2)
            # inner_neg_noisy = - torch.sum(pcl_diff_exp_noisy * gramian_feat ) #, dim=(1,2)
            # inner_neg = inner_neg - inner_neg_noisy
        else:
            if self.color_in_cost:
                inner_neg = - torch.sum(pcl_diff_exp * gramian_feat * gramian_color )
            else:
                inner_neg = - torch.sum(pcl_diff_exp * gramian_feat ) 

            final_loss = inner_neg
            if self.sparse_mode:
                final_loss = inner_neg + fea_norm_sum
        
        if pose1_2_pred is not None:
            xyz2_trans_pred = torch.matmul(pose1_2_pred, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred, self.dist_coef)
            if self.color_in_cost:
                inner_neg_pred = - torch.sum(pcl_diff_exp_pred * gramian_feat * gramian_color )
            else:
                inner_neg_pred = - torch.sum(pcl_diff_exp_pred * gramian_feat ) 

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