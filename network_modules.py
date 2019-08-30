import torch
import torch.nn as nn
from unet import UNet
from unet_pose_regressor import UNetRegressor
import numpy as np
from dataloader import pose_from_euler_t, pose_from_euler_t_Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from geometry_plot import draw3DPts

from geometry import kern_mat, gramian, gen_3D

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
        fx=48,
        fy=48,
        cx=48,
        cy=36, 
        diff_mode=True,
        sparse_mode=False,
        kernalize=False,
        color_in_cost=False, 
        L2_norm=True
    ):
        super(UNetInnerProd, self).__init__()
        self.device = device
        self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)
        self.model_loss = innerProdLoss(device, fx, fy, cx, cy, diff_mode, sparse_mode, kernalize, color_in_cost, L2_norm).to(device)
        self.pose_predictor = UNetRegressor(batch_norm=True, feature_channels=n_classes*2).to(device)

    def forward(self, img1, img2, dep1, dep2, idep1, idep2, pose1_2):
        feature1 = self.model_UNet(img1)
        feature2 = self.model_UNet(img2)

        dep1.requires_grad = False
        dep2.requires_grad = False
        pose1_2.requires_grad = False

        # with pose predictor
        euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
        pose1_2_pred = pose_from_euler_t_Tensor(euler_pred, device=self.device)

        loss, innerp_loss, feat_norm, innerp_loss_pred = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2, pose1_2_pred)
        return feature1, feature2, loss, innerp_loss, feat_norm, innerp_loss_pred, euler_pred

        # # without pose predictor
        # loss, innerp_loss, feat_norm = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2)
        # return feature1, feature2, loss, innerp_loss, feat_norm



class innerProdLoss(nn.Module):
    def __init__(self, device, fx=48, fy=48, cx=48, cy=36, diff_mode=True, sparse_mode=False, kernalize=False, color_in_cost=False, L2_norm=True):
        super(innerProdLoss, self).__init__()
        self.device = device
        height = int(2*cy)
        width = 2*cx
        # K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
        K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ])
        
        inv_K = torch.Tensor(np.linalg.inv(K)).to(self.device)
        K = torch.Tensor(K).to(self.device)

        u_grid = torch.Tensor(np.arange(width) )
        v_grid = torch.Tensor(np.arange(height) )
        uu_grid = u_grid.unsqueeze(0).expand((height, -1) ).reshape(-1)
        vv_grid = v_grid.unsqueeze(1).expand((-1, width) ).reshape(-1)
        uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N
        self.yz1_grid = torch.mm(inv_K, uv1_grid).to(self.device) # 3*N
        self.diff_mode = diff_mode
        self.sparse_mode = sparse_mode
        self.kernalize = kernalize
        self.color_in_cost = color_in_cost
        self.L2_norm = L2_norm

    def gen_rand_pose(self):
        trans_noise = np.random.normal(scale=1.0, size=(3,))
        rot_noise = np.random.normal(scale=5.0, size=(3,))
        self.pose1_2_noise = pose_from_euler_t(trans_noise[0], 3*trans_noise[1], trans_noise[2], 0, rot_noise[1], 3*rot_noise[2])
        self.pose1_2_noise = torch.Tensor(self.pose1_2_noise).to(self.device)
        self.pose1_2_noise.requires_grad = False

    def forward(self, feature1, feature2, depth1, depth2, pose1_2, img1, img2, pose1_2_pred=None): # img1 and img2 only for visualization
        xyz1, xyz2 = gen_3D(self.yz1_grid, depth1, depth2)
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans = torch.matmul(pose1_2, xyz2_homo)[:, 0:3, :] # B*3*N

        if self.sparse_mode:
            normalization_mode = 0
        else:
            normalization_mode = 1
        gramian_feat, fea_norm_sum = gramian(feature1, feature2, normalization_mode, kernalize=self.kernalize, L2_norm=self.L2_norm)
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans)

        if self.color_in_cost:
            gramian_color, _ = gramian(img1, img2, norm_mode=0, kernalize=self.kernalize, L2_norm=False)

        # n_pts_1 = img1.shape[2]*img1.shape[3]
        # n_pts_2 = img2.shape[2]*img2.shape[3]

        # img1_flat = img1.reshape(-1, 3, n_pts_1)
        # img2_flat = img2.reshape(-1, 3, n_pts_2)
        
        # draw3DPts(xyz1, xyz2_trans, img1_flat, img2_flat)
        # draw3DPts(xyz1, xyz2, img1_flat, img2_flat)
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
            pose1_2_noisy = torch.matmul(pose1_2, self.pose1_2_noise)
            xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_noisy = kern_mat(xyz1, xyz2_trans_noisy)

            pcl_diff_exp_diff = pcl_diff_exp - pcl_diff_exp_noisy

            if self.color_in_cost:
                inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat * gramian_color )
            else:
                inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat )

            final_loss = inner_neg
            if self.sparse_mode:
                final_loss = inner_neg + fea_norm_sum

            # draw3DPts(xyz1, xyz2_trans_noisy, img1_flat, img2_flat)

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
            pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred)
            if self.color_in_cost:
                inner_neg_pred = - torch.sum(pcl_diff_exp_pred * gramian_feat * gramian_color )
            else:
                inner_neg_pred = - torch.sum(pcl_diff_exp_pred * gramian_feat ) 

            final_loss = final_loss + inner_neg_pred

            
            return final_loss, inner_neg, fea_norm_sum, inner_neg_pred

        return final_loss, inner_neg, fea_norm_sum
