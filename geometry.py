import torch
import torch.nn as nn
from unet import UNet
import numpy as np
from dataloader import pose_from_euler_t

def kern_mat(pcl_1, pcl_2):
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
    pcl_1_expand = pcl_1.unsqueeze(-1).expand(B, C, N1, N2)
    pcl_2_expand = pcl_2.unsqueeze(-2).expand(B, C, N1, N2)
    pcl_diff_exp = torch.exp(-torch.norm(pcl_1_expand - pcl_2_expand, dim=1)/1e-2  ) # B*N1*N2 

    return pcl_diff_exp

def gramian(feature1, feature2):
    """inputs are B*C*H*W tensors, C corresponding to feature dimension (default 3)
    output is B*N1*N2
    """

    batch_size = feature1.shape[0]
    channels = feature1.shape[1]
    n_pts_1 = feature1.shape[2]*feature1.shape[3]
    n_pts_2 = feature2.shape[2]*feature2.shape[3]

    fea_flat_1 = feature1.reshape(batch_size, channels, n_pts_1) # B*C*N1
    fea_flat_2 = feature2.reshape(batch_size, channels, n_pts_2) # B*C*N2

    fea_norm_1 = torch.norm(fea_flat_1, dim=1)
    fea_norm_2 = torch.norm(fea_flat_2, dim=1)
    
    fea_flat_1 = torch.div(fea_flat_1, fea_norm_1.expand_as(fea_flat_1) )
    fea_flat_2 = torch.div(fea_flat_2, fea_norm_2.expand_as(fea_flat_2) )
    
    
    gramian = torch.matmul(fea_flat_1.transpose(1,2), fea_flat_2)
    return gramian

def gen_3D(xy1_grid, depth1, depth2):
    """
    depth1/2 are B*1*H*W, xy1_grid is 3*N(N=H*W)
    transform both to B*C(1 or 3)*N
    both outputs are B*3*N
    """
    batch_size = depth1.shape[0]
    n_pts_1 = depth1.shape[2]*depth1.shape[3]
    n_pts_2 = depth2.shape[2]*depth2.shape[3]

    depth1_flat = depth1.reshape(-1, 1, n_pts_1).expand(-1, 3, -1) # B*3*N
    depth2_flat = depth2.reshape(-1, 1, n_pts_2).expand(-1, 3, -1)
    xy1_grid_batch = xy1_grid.expand(batch_size, -1, -1) # B*3*N
    
    xyz_1 = xy1_grid_batch * depth1_flat
    xyz_2 = xy1_grid_batch * depth2_flat

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
        cy=36
    ):
        super(UNetInnerProd, self).__init__()
        self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)
        self.model_loss = innerProdLoss(device, fx, fy, cx, cy).to(device)

    def forward(self, img1, img2, dep1, dep2, pose1_2):
        feature1 = self.model_UNet(img1)
        feature2 = self.model_UNet(img2)

        dep1.requires_grad = False
        dep2.requires_grad = False

        loss = self.model_loss(feature1, feature2, dep1, dep2, pose1_2)

        return feature1, feature2, loss


class innerProdLoss(nn.Module):
    def __init__(self, device, fx=48, fy=48, cx=48, cy=36):
        super(innerProdLoss, self).__init__()
        self.device = device
        height = int(2*cy)
        width = 2*cx
        K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
        inv_K = torch.Tensor(np.linalg.inv(K)).to(self.device)
        K = torch.Tensor(K).to(self.device)

        u_grid = torch.Tensor(np.arange(width) )
        v_grid = torch.Tensor(np.arange(height) )
        uu_grid = u_grid.unsqueeze(0).expand((height, -1) ).reshape(-1)
        vv_grid = v_grid.unsqueeze(1).expand((-1, width) ).reshape(-1)
        uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N
        self.xy1_grid = torch.mm(inv_K, uv1_grid).to(self.device) # 3*N



    def forward(self, feature1, feature2, depth1, depth2, pose1_2):
        xyz1, xyz2 = gen_3D(self.xy1_grid, depth1, depth2)
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans = torch.matmul(pose1_2, xyz2_homo)[:, 0:3, :] # B*3*N

        gramian_feat = gramian(feature1, feature2)
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans)

        diff_mode = False
        if diff_mode:
            trans_noise = np.random.normal(scale=1.0, size=(3,))
            rot_noise = np.random.normal(scale=3.0, size=(3,))
            pose1_2_noise = pose_from_euler_t(trans_noise[0], trans_noise[1], trans_noise[2], rot_noise[0], rot_noise[1], rot_noise[2])
            pose1_2_noise = torch.Tensor(pose1_2_noise).to(self.device)
            pose1_2_noisy = torch.matmul(pose1_2, pose1_2_noise)
            xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_noisy = kern_mat(xyz1, xyz2_trans_noisy)

            pcl_diff_exp_diff = pcl_diff_exp - pcl_diff_exp_noisy
            inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat )

            # inner_neg = - torch.sum(pcl_diff_exp * gramian_feat ) #, dim=(1,2)
            # inner_neg_noisy = - torch.sum(pcl_diff_exp_noisy * gramian_feat ) #, dim=(1,2)
            # inner_neg = inner_neg - inner_neg_noisy
        else:
            inner_neg = - torch.sum(pcl_diff_exp * gramian_feat )


        return inner_neg