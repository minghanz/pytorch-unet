import torch

# import sub_cuda
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

        pcl_diff = SubNormFunction.apply(pcl_1.contiguous(), pcl_2.contiguous())
        # print(pcl_diff.device)
        assert not torch.isnan(pcl_diff).any()
        assert not torch.isinf(pcl_diff).any()
        # pcl_diff_exp = torch.exp(-pcl_diff * dist_coef)
        pcl_diff_exp = torch.exp(-pcl_diff / (2 * dist_coef * dist_coef) )

        # pcl_1_expand = pcl_1.unsqueeze(-1).expand(B, C, N1, N2)
        # pcl_2_expand = pcl_2.unsqueeze(-2).expand(B, C, N1, N2)
        # pcl_diff_exp = torch.exp(-torch.norm(pcl_1_expand - pcl_2_expand, dim=1) * dist_coef  ) # B*N1*N2 

    return pcl_diff_exp

def gramian(fea_flat_1, fea_flat_2, norm_mode, kernalize, norm_dim):
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
        gramian = kern_mat(fea_flat_1, fea_flat_2, dist_coef=1e0) # , dist_coef=1e1

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


