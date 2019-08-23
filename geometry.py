import torch
import torch.nn as nn
from unet import UNet
from unet_pose_regressor import UNetRegressor
import numpy as np
from dataloader import pose_from_euler_t, pose_from_euler_t_Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from geometry_plot import draw3DPts

import sub_cuda
import sub_norm_cuda
from torch.autograd import Function

class SubNormFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = sub_norm_cuda.forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return outputs

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors
        dx1, dx2 = sub_norm_cuda.backward(dy.to(torch.device('cuda:1')), x1.to(torch.device('cuda:1')), x2.to(torch.device('cuda:1')))
        return dx1.to(torch.device('cuda:1')), dx2.to(torch.device('cuda:1'))

# class SubFunction(Function):
#     @staticmethod
#     def forward(ctx, x1, x2):
#         outputs = sub_cuda.forward(x1, x2 )
#         return outputs

#     @staticmethod
#     def backward(ctx, dy):
#         dx1, dx2 = sub_cuda.backward(dy.to(torch.device('cuda:1')) )
#         return dx1.to(torch.device('cuda:1')), dx2.to(torch.device('cuda:1')) 

def kern_mat(pcl_1, pcl_2, dist_coef=1e-1):
    """
    pcl1 and pcl2 are B*3*N tensors, 3 for x, y, z
    output is B*N1*N2
    """
    input_size_1 = list(pcl_1.size() )
    input_size_2 = list(pcl_2.size() )
    B = input_size_1[0]
    C = input_size_1[1]
    N1 = input_size_1[2]
    N2 = input_size_2[2]

    print("kern_mat")
    if False : # N1%4 ==0 and N2%2==0 
        N1_split = int(N1/4)
        N2_split = int(N2/2)
        pcl_2_expand_1 = pcl_2[:,:,0:N2_split].unsqueeze(-2).expand(B, C, N1_split, N2_split)
        pcl_2_expand_2 = pcl_2[:,:,N2_split:2*N2_split].unsqueeze(-2).expand(B, C, N1_split, N2_split)

        pcl_1_expand_1 = pcl_1[:,:,0:N1_split].unsqueeze(-1).expand(B, C, N1_split, N2_split)
        pcl_1_expand_2 = pcl_1[:,:,N1_split:2*N1_split].unsqueeze(-1).expand(B, C, N1_split, N2_split)
        pcl_1_expand_3 = pcl_1[:,:,2*N1_split:3*N1_split].unsqueeze(-1).expand(B, C, N1_split, N2_split)
        pcl_1_expand_4 = pcl_1[:,:,3*N1_split:4*N1_split].unsqueeze(-1).expand(B, C, N1_split, N2_split)

        pcl_diff_exp_11 = torch.exp(-torch.norm(pcl_1_expand_1 - pcl_2_expand_1, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_11 = torch.exp(-torch.norm(pcl_1_expand_1 - pcl_2_expand_1, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_21 = torch.exp(-torch.norm(pcl_1_expand_2 - pcl_2_expand_1, dim=1) * dist_coef  ) # B*N1*N2 
        
        pcl_diff_exp_31 = torch.exp(-torch.norm(pcl_1_expand_3 - pcl_2_expand_1, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_41 = torch.exp(-torch.norm(pcl_1_expand_4 - pcl_2_expand_1, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_12 = torch.exp(-torch.norm(pcl_1_expand_1 - pcl_2_expand_2, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_22 = torch.exp(-torch.norm(pcl_1_expand_2 - pcl_2_expand_2, dim=1) * dist_coef  ) # B*N1*N2 
        
        pcl_diff_exp_32 = torch.exp(-torch.norm(pcl_1_expand_3 - pcl_2_expand_2, dim=1) * dist_coef  ) # B*N1*N2 

        pcl_diff_exp_42 = torch.exp(-torch.norm(pcl_1_expand_4 - pcl_2_expand_2, dim=1) * dist_coef  ) # B*N1*N2 
        
        pcl_diff_exp_1 = torch.cat((pcl_diff_exp_11, pcl_diff_exp_21, pcl_diff_exp_31, pcl_diff_exp_41), dim=1)
        pcl_diff_exp_2 = torch.cat((pcl_diff_exp_12, pcl_diff_exp_22, pcl_diff_exp_32, pcl_diff_exp_42), dim=1)
        
        pcl_diff_exp = torch.cat((pcl_diff_exp_1, pcl_diff_exp_2), dim=2)

    else:
        # pcl_1_expand = pcl_1.unsqueeze(-1)
        # print(pcl_1_expand.shape)
        # pcl_2_expand = pcl_2.unsqueeze(-2)
        # print(pcl_2_expand.shape)
        # pcl_diff = pcl_1_expand - pcl_2_expand
        
        # pcl_diff = SubFunction.apply(pcl_1, pcl_2.contiguous()).to(torch.device('cuda:1'))
        # pcl_diff_exp = torch.exp(-torch.norm(pcl_diff, dim=1) * dist_coef  )

        pcl_diff = SubNormFunction.apply(pcl_1, pcl_2.contiguous()).to(torch.device('cuda:1'))
        pcl_diff_exp = torch.exp(-pcl_diff * dist_coef)

        # pcl_1_expand = pcl_1.unsqueeze(-1).expand(B, C, N1, N2)
        # pcl_2_expand = pcl_2.unsqueeze(-2).expand(B, C, N1, N2)
        # pcl_diff_exp = torch.exp(-torch.norm(pcl_1_expand - pcl_2_expand, dim=1) * dist_coef  ) # B*N1*N2 

    return pcl_diff_exp

def gramian(feature1, feature2, norm_mode, kernalize, L2_norm):
    """inputs are B*C*H*W tensors, C corresponding to feature dimension (default 3)
    output is B*N1*N2
    """

    batch_size = feature1.shape[0]
    channels = feature1.shape[1]
    n_pts_1 = feature1.shape[2]*feature1.shape[3]
    n_pts_2 = feature2.shape[2]*feature2.shape[3]

    fea_flat_1 = feature1.reshape(batch_size, channels, n_pts_1) # B*C*N1
    fea_flat_2 = feature2.reshape(batch_size, channels, n_pts_2) # B*C*N2

    if L2_norm:
        fea_norm_1 = torch.norm(fea_flat_1, dim=1)
        fea_norm_2 = torch.norm(fea_flat_2, dim=1)

        fea_norm_sum_1 = torch.sum(fea_norm_1)
        fea_norm_sum_2 = torch.sum(fea_norm_2)
    else:
        fea_norm_sum_1 = torch.sum(torch.abs(fea_flat_1))
        fea_norm_sum_2 = torch.sum(torch.abs(fea_flat_2))
    
    if norm_mode == 1:
        fea_flat_1 = torch.div(fea_flat_1, fea_norm_1.expand_as(fea_flat_1) ) # +1e-6 applied if feature is non-positive
        fea_flat_2 = torch.div(fea_flat_2, fea_norm_2.expand_as(fea_flat_2) ) # +1e-6 applied if feature is non-positive
    elif norm_mode == 2:
        fea_flat_1 = fea_flat_1 /100.0
        fea_flat_2 = fea_flat_2 /100.0
    
    if not kernalize:
        gramian = torch.matmul(fea_flat_1.transpose(1,2), fea_flat_2) 
    else:
        gramian = kern_mat(fea_flat_1, fea_flat_2, dist_coef=1e1)

    return gramian, fea_norm_sum_1 + fea_norm_sum_2

def gen_3D(yz1_grid, depth1, depth2):
    """
    depth1/2 are B*1*H*W, yz1_grid is 3*N(N=H*W)
    transform both to B*C(1 or 3)*N
    both outputs are B*3*N
    """
    batch_size = depth1.shape[0]
    n_pts_1 = depth1.shape[2]*depth1.shape[3]
    n_pts_2 = depth2.shape[2]*depth2.shape[3]

    depth1_flat = depth1.reshape(-1, 1, n_pts_1).expand(-1, 3, -1) # B*3*N
    depth2_flat = depth2.reshape(-1, 1, n_pts_2).expand(-1, 3, -1)
    yz1_grid_batch = yz1_grid.expand(batch_size, -1, -1) # B*3*N
    
    xyz_1 = yz1_grid_batch * depth1_flat
    xyz_2 = yz1_grid_batch * depth2_flat

    return xyz_1, xyz_2

    
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
        self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)
        self.model_loss = innerProdLoss(device, fx, fy, cx, cy, diff_mode, sparse_mode, kernalize, color_in_cost, L2_norm).to(device)
        self.pose_predictor = UNetRegressor(batch_norm=True, feature_channels=n_classes)

    def forward(self, img1, img2, dep1, dep2, idep1, idep2, pose1_2):
        feature1 = self.model_UNet(img1)
        feature2 = self.model_UNet(img2)

        dep1.requires_grad = False
        dep2.requires_grad = False
        pose1_2.requires_grad = False

        euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
        pose1_2_pred = pose_from_euler_t_Tensor(euler_pred)

        print("model_loss begin")
        loss, innerp_loss, feat_norm, innerp_loss_pred = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2, pose1_2_pred)
        print("model_loss end")

        return feature1, feature2, loss, innerp_loss, feat_norm, innerp_loss_pred


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
                fea_norm_sum = fea_norm_sum *1e3
            else: 
                fea_norm_sum = fea_norm_sum *1e4

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

