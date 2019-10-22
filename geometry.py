import torch

# import sub_cuda
import sub_norm_cuda
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

class SubNormFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        outputs = sub_norm_cuda.forward(x1, x2)
        ctx.save_for_backward(x1, x2)
        return outputs

    @staticmethod
    def backward(ctx, dy):
        x1, x2 = ctx.saved_tensors

        dx1, dx2 = sub_norm_cuda.backward(dy, x1, x2)
        return dx1, dx2

# class SubFunction(Function):
#     @staticmethod
#     def forward(ctx, x1, x2):
#         outputs = sub_cuda.forward(x1, x2 )
#         return outputs

#     @staticmethod
#     def backward(ctx, dy):
#         dx1, dx2 = sub_cuda.backward(dy.to(torch.device('cuda')) )
#         return dx1.to(torch.device('cuda')), dx2.to(torch.device('cuda')) 

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

    # print("kern_mat")
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
        
        # pcl_diff = SubFunction.apply(pcl_1, pcl_2.contiguous()).to(torch.device('cuda'))
        # pcl_diff_exp = torch.exp(-torch.norm(pcl_diff, dim=1) * dist_coef  )

        # print(pcl_1.shape, pcl_2.shape)
        pcl_diff = SubNormFunction.apply(pcl_1.contiguous(), pcl_2.contiguous())
        assert not torch.isnan(pcl_diff).any()
        assert not torch.isinf(pcl_diff).any()
        # pcl_diff_exp = torch.exp(-pcl_diff * dist_coef)
        pcl_diff_exp = torch.exp(-pcl_diff / (2 * dist_coef * dist_coef) )

        # pcl_1_expand = pcl_1.unsqueeze(-1).expand(B, C, N1, N2)
        # pcl_2_expand = pcl_2.unsqueeze(-2).expand(B, C, N1, N2)
        # pcl_diff_exp = torch.exp(-torch.norm(pcl_1_expand - pcl_2_expand, dim=1) * dist_coef  ) # B*N1*N2 

    return pcl_diff_exp

def gramian(fea_flat_1, fea_flat_2, norm_mode, kernalize, norm_dim, dist_coef=1e0):
    """
    RBF inner product or normal inner product, for features.
    inputs are B*C*N tensors, C corresponding to feature dimension (default 3)
    output is B*N1*N2
    norm_dim: 0 (do not calculate norm), 1 or 2
    norm_mode: whether to normalize the feature map
    """

    fea_norm_sum_1 = torch.tensor(0., dtype=fea_flat_1.dtype, device=fea_flat_1.get_device())
    fea_norm_sum_2 = torch.tensor(0., dtype=fea_flat_2.dtype, device=fea_flat_2.get_device())

    if norm_dim == 1:
        ### preserve pixel dimension
        fea_norm_1 = torch.norm(fea_flat_1, dim=1, keepdim=True) #L2 norm across channels
        fea_norm_2 = torch.norm(fea_flat_2, dim=1, keepdim=True)
    elif norm_dim == 2:
        ### preserve channel dimension
        fea_norm_1 = torch.mean(torch.abs(fea_flat_1), dim=2, keepdim=True) # L1 norm across pixels
        fea_norm_2 = torch.mean(torch.abs(fea_flat_2), dim=2, keepdim=True)
    
    if norm_mode == True:
        # normalize the feature map using the above norm
        fea_flat_1 = torch.div(fea_flat_1, fea_norm_1 ) # +1e-6 applied if feature is non-positive
        fea_flat_2 = torch.div(fea_flat_2, fea_norm_2 ) # +1e-6 applied if feature is non-positive

        ### normally speaking, no need to calculate norm_sum again since the feature map is already normalized
        ### but we calculate L2 norm across pixels here to measure the sparsity
        if norm_dim == 2:
            ## L2 norm across pixels, indicate the sparsity across pixels
            fea_norm_sum_1 = - torch.mean( torch.norm(fea_flat_1, dim=2) )
            fea_norm_sum_2 = - torch.mean( torch.norm(fea_flat_2, dim=2) )
    else:
        if norm_dim == 1 or norm_dim == 2:
            fea_norm_sum_1 = torch.mean(fea_norm_1)
            fea_norm_sum_2 = torch.mean(fea_norm_2)

        
    
    if not kernalize:
        gramian = torch.matmul(fea_flat_1.transpose(1,2), fea_flat_2) 
    else:
        gramian = kern_mat(fea_flat_1, fea_flat_2, dist_coef=dist_coef) # , dist_coef=1e1

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

def gen_3D_flat(yz1_grid, depth1, depth2):
    """
    depth1/2 are B*1*N, yz1_grid is 3*N(N=H*W)
    transform both to B*C(1 or 3)*N
    both outputs are B*3*N
    """
    batch_size = depth1.shape[0]

    depth1_flat = depth1.expand(-1, 3, -1) # B*3*N
    depth2_flat = depth2.expand(-1, 3, -1)
    yz1_grid_batch = yz1_grid.expand(batch_size, -1, -1) # B*3*N
    
    xyz_1 = yz1_grid_batch * depth1_flat
    xyz_2 = yz1_grid_batch * depth2_flat

    return xyz_1, xyz_2

def fill_partial_depth(depth, value=100):
    dummy_depth = torch.ones_like(depth) * value
    depth = torch.where(depth > 0, depth, dummy_depth)
    return depth

def reproject_image(host_xyz_on_guest, K, guest_image, zdim=2):
    """
    Reconstruct host frame using host depth, transformation T and guest image. 
    First, reproject host image coord grid onto guest image frame using host_depth and T
    Second, reconstruct host frame by interpolation on guest image
    host_xyz_on_guest: B*3*N
    K: 3*3
    zdim: 2 for TUM, 0 for CARLA
    guest_image: B*C*H*W
    warped_image: B*C*H*W
    """
    width = guest_image.shape[3]
    height = guest_image.shape[2]

    ## turn xyz to xy1 B*3*N (3: x, y, z)
    host_xy_on_guest = host_xyz_on_guest / host_xyz_on_guest[:, zdim:zdim+1, :].expand_as(host_xyz_on_guest)
    ## turn xy1 to uv1 B*2*N (2: u, v)
    host_uv_on_guest = torch.matmul(K, host_xy_on_guest)[:, :2, :]
    ## turn to B*H*W*2 (2: u, v)
    host_uv_square = host_uv_on_guest.reshape( host_uv_on_guest.shape[0], 2, height, width ).permute(0, 2, 3, 1)
    ## normalize to [-1, 1]
    host_uv_square[..., 0] /= (width-1)
    host_uv_square[..., 1] /= (height-1)
    host_uv_square = (host_uv_square - 0.5) * 2
    ## reconstruct host image by sampling on guest image
    warped_image = F.grid_sample( guest_image, host_uv_square, padding_mode="border")

    return warped_image

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def rgb_to_hsv(image, flat=False):
    """Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.
        flat: True if input B*C*N, False if input B*C*H*W

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if not flat:
        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W) given flat=False. Got {}"
                            .format(image.shape))
    else:
        if len(image.shape) < 2 or image.shape[-2] != 3:
            raise ValueError("Input size must have a shape of (*, 3, N) given flat=True. Got {}"
                            .format(image.shape))

    if not flat:
        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        maxc: torch.Tensor = image.max(-3)[0]
        minc: torch.Tensor = image.min(-3)[0]
    else:

        r: torch.Tensor = image[..., 0, :]
        g: torch.Tensor = image[..., 1, :]
        b: torch.Tensor = image[..., 2, :]

        maxc: torch.Tensor = image.max(-2)[0]
        minc: torch.Tensor = image.min(-2)[0]

    v: torch.Tensor = maxc  # brightness
    
    # ZMH: avoid division by zero
    v = torch.where(
        v == 0, torch.ones_like(v)*1e-3, v)

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v  # saturation

    # avoid division by zero
    deltac: torch.Tensor = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg]: torch.Tensor = 2.0 + rc[maxg] - bc[maxg]
    h[maxr]: torch.Tensor = bc[maxr] - gc[maxr]
    h[minc == maxc]: torch.Tensor = 0.0

    h: torch.Tensor = (h / 6.0) % 1.0

    if not flat:
        return torch.stack([h, s, v], dim=-3)
    else:
        return torch.stack([h, s, v], dim=-2)



def gen_rand_pose(source, noise_trans_scale, device):
    ### do not use np.random to generate random number because they can not be tracked by pytorch backend
    # trans_noise = np.random.normal(scale=self.noise_trans_scale, size=(3,))
    # rot_noise = np.random.normal(scale=3.0, size=(3,))
    #####################################################################################################
    trans_noise = torch.randn((3), dtype=torch.float, device=device) * noise_trans_scale
    # rot_noise = torch.randn((3), dtype=torch.float, device=self.device) * 3.0
    rot_noise = torch.zeros(3, dtype=torch.float, device=device)
    
    if source=='CARLA':
        noise_euler_tensor = torch.stack([ trans_noise[0], 3*trans_noise[1], trans_noise[2], 0, rot_noise[1], 3*rot_noise[2] ]).unsqueeze(0)
        pose1_2_noise = pose_from_euler_t_Tensor(noise_euler_tensor, device=device).squeeze(0)
    elif source=='TUM':
        noise_euler_tensor = torch.stack([trans_noise[0], trans_noise[1], trans_noise[2], rot_noise[0], rot_noise[1], rot_noise[2]]).unsqueeze(0)
        pose1_2_noise = pose_from_euler_t_Tensor(noise_euler_tensor, device=device).squeeze(0)
    pose1_2_noise.requires_grad = False

    return pose1_2_noise

def gen_noisy_pose(pose1_2_noise, mean_trans_2):
    ##### move ot center, purturb, rotate back
    device = mean_trans_2.device

    trans = torch.tensor([0., 0., 0.], dtype=torch.float, device=device)

    trans_rot = torch.stack([-mean_trans_2[0], -mean_trans_2[1], -mean_trans_2[2]])
    trans_euler = torch.cat((trans_rot, trans), dim=0 ).unsqueeze(0)
    trans_back = pose_from_euler_t_Tensor(trans_euler, device=device).squeeze(0)
    trans_back.requires_grad = False

    trans_rot = torch.stack([mean_trans_2[0], mean_trans_2[1], mean_trans_2[2]])
    trans_euler = torch.cat((trans_rot, trans), dim=0 ).unsqueeze(0)
    trans_forth = pose_from_euler_t_Tensor(trans_euler, device=device).squeeze(0)
    trans_forth.requires_grad = False

    pose1_2_noisy = torch.matmul(trans_forth, torch.matmul(pose1_2_noise, trans_back) )
    return pose1_2_noisy

def gen_uvgrid(width, height, inv_K):
    device = inv_K.device
    u_grid = torch.tensor(np.arange(width), dtype=torch.float, device=device )
    v_grid = torch.tensor(np.arange(height), dtype=torch.float, device=device )
    uu_grid = u_grid.unsqueeze(0).expand((height, -1) ).reshape(-1)
    vv_grid = v_grid.unsqueeze(1).expand((-1, width) ).reshape(-1)
    uv1_grid = torch.stack( (uu_grid, vv_grid, torch.ones(uu_grid.size(), dtype=torch.float, device=device) ), dim=0 ) # 3*N, correspond to u,v,1
    yz1_grid = torch.mm(inv_K, uv1_grid) # 3*N, corresponding to x,y,z
    return uv1_grid, yz1_grid

# def gen_cam_K(source, width, height):
#     '''
#         CARLA and TUM have differenct definition of relation between xyz coordinate and uv coordinate.
#         CARLA xyz is front-right(u)-down(v)(originally up, which is left handed, fixed to down in pose_from_euler_t function)
#         TUM xyz is right(u)-down(v)-front
#         Output is nparray
#     '''
#     assert source == 'CARLA' or source == 'TUM', 'source unrecognized'
#     if source == 'CARLA':
#         fx=int(width/2)
#         fy=int(width/2)
#         cx=int(width/2)
#         cy=int(height/2)
#         K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
#     elif source == 'TUM':
#         #### see: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
#         fx = width/640.0*525.0  # focal length x
#         fy = height/480.0*525.0  # focal length y
#         cx = width/640.0*319.5  # optical center x
#         cy = height/480.0*239.5  # optical center y
#         K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
#     return K