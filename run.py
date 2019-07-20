import torch
import torch.nn.functional as F
from unet import UNet
from dataloader import ImgPoseDataset, ToTensor, Rescale

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import torch.autograd.Function as Function
import torch.nn as nn
import numpy as np

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
    pcl_diff_exp = torch.exp(torch.norm(pcl_1_expand - pcl_2_expand, dim=1) ) # B*N1*N2 

    print(pcl_diff_exp.shape)

    return pcl_diff_exp

def gramian(feature1, feature2):
    """inputs are B*C*H*W tensors, C corresponding to feature dimension (default 3)
    output is B*N1*N2
    """
    print(feature1.shape)
    print(feature2.shape)

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
    print(fea_flat_1.shape)
    print(fea_flat_2.shape)
    
    
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
    
# class innerProdLossFunc(Function):

#     @staticmethod
#     def forward(ctx, feature1, feature2, depth1, depth2, pose1_2, xy1_grid):
        

    


class innerProdLoss(nn.Module):
    def __init__(self, device, fx=56, fy=56, cx=56, cy=42):
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

        pcl_diff_exp = kern_mat(xyz1, xyz2_trans)
        gramian_feat = gramian(feature1, feature2)

        inner_neg = - torch.sum(pcl_diff_exp * gramian_feat ) #, dim=(1,2)

        return inner_neg


def main():


    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

    model = UNet(in_channels=3, n_classes=3, depth=3, wf=4, padding=True).to(device)
    optim = torch.optim.Adam(model.parameters())
    img_pose_dataset = ImgPoseDataset(transform=transforms.Compose([Rescale(), ToTensor(device=device) ]) )

    data_to_load = DataLoader(img_pose_dataset, batch_size=3, shuffle=True)

    loss_model = innerProdLoss(device=device).to(device)

    epochs = 10

    print('going into loop')
    for _ in range(epochs):
        for i_batch, sample_batch in enumerate(data_to_load):
            img1 = sample_batch['image 1']
            img2 = sample_batch['image 2']
            dep1 = sample_batch['depth 1']
            dep2 = sample_batch['depth 2']
            pose1_2 = sample_batch['rela_pose']
            
            feature1 = model(img1)
            feature2 = model(img2)
            dep1.requires_grad = False
            dep2.requires_grad = False
            

            loss = loss_model(feature1, feature2, dep1, dep2, pose1_2)

            optim.zero_grad()
            loss.backward()
            optim.step()

if __name__ == "__main__":
    main()