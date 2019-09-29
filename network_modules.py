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

# from fastai.vision import *
# from torchvision import models
import segmentation_models_pytorch as smp

# from run import feat_svd

def feat_svd(feature):
    # input feature is b*c*h*w, need to change to b*c*n
    b = feature.shape[0]
    c = feature.shape[1]
    h = feature.shape[2]
    w = feature.shape[3]
    feat_flat = feature.reshape(b, c, h*w)
    
    # feat_mean = torch.mean(feature, dim=(2,3), keepdim=False)
    # feat_flat = feat_flat - feat_mean.unsqueeze(2).expand_as(feat_flat)
    # print(feat_flat.device)
    # print(feat_flat.shape)

    feat_new = feat_svd_flat(feat_flat)

    feat_img = feat_new.reshape(b,c,h,w)
    feat_img_3 = feat_img[:,0:3,:,:]
    return feat_img_3, feat_img

def feat_svd_flat(feat_flat):
    feat_mean = feat_flat.mean(dim=2, keepdim=True)
    feat_flat = feat_flat - feat_mean.expand_as(feat_flat)
    u, s, v = torch.svd(feat_flat)
    # feat_new = torch.bmm( u[:,:,0:3].transpose(1,2), feat_flat) # b*3*n
    # feat_img = feat_new.reshape(b,3,h,w)
    feat_new = torch.bmm( u.transpose(1,2), feat_flat) # b*3*n

    return feat_new

def feat_svd_flat_single(feat_flat):
    feat_mean = feat_flat.mean(dim=1, keepdim=True)
    feat_flat = feat_flat - feat_mean.expand_as(feat_flat)
    u, s, v = torch.svd(feat_flat)
    feat_new = torch.mm( u.transpose(0,1), feat_flat) # b*3*n
    return feat_new

def model_init(opt_unet, device):
    if opt_unet.pretrained_mode:
        def final_act(x):
            return 0.05 * x
        def sigmoid_scaled(x):
            sig = nn.Sigmoid()
            x = 2 * sig(x*0.4) - 1
            return x

        if opt_unet.weight_map_mode:
            model_UNet = smp.UnetWithLastOut('resnet34', encoder_weights='imagenet', classes=opt_unet.n_classes, activation=sigmoid_scaled ).to(device)
        else: # 2/3/4
            model_UNet = smp.Unet('resnet34', encoder_weights='imagenet', classes=opt_unet.n_classes, activation=final_act ).to(device)

        for param in model_UNet.encoder.parameters():
            param.requires_grad = False 
        # pretrained_model = models.resnet34(pretrained=True)
        # self.model_Unet = models.unet.DynamicUnet(encoder=pretrained_model, n_classes=n_classes).to(device)
    else:
        model_UNet = UNet(opt_unet.in_channels, opt_unet.n_classes, opt_unet.depth, opt_unet.wf, opt_unet.padding, opt_unet.batch_norm, opt_unet.up_mode).to(device)

    return model_UNet

class UNetInnerProd(nn.Module):
    def __init__(self,
        loss_options,
        unet_options,
        device= torch.device('cuda')
    ):
        super(UNetInnerProd, self).__init__()
        self.device = device
        self.opt_loss = loss_options
        self.opt_unet = unet_options

        self.model_UNet = model_init(self.opt_unet, self.device)
        # self.model_init(self.opt_unet, self.device)

        self.model_loss = innerProdLoss(device, self.opt_loss).to(device)
        
        if self.opt_unet.pose_predict_mode:
            self.pose_predictor = UNetRegressor(batch_norm=True, feature_channels=n_classes*2).to(device)

    def forward(self, sample_batch):
        img1 = sample_batch['image 1']
        img2 = sample_batch['image 2']
        img1_raw = sample_batch['image 1 raw']
        img2_raw = sample_batch['image 2 raw']
        
        dep1 = sample_batch['depth 1']
        dep2 = sample_batch['depth 2']
        idep1 = sample_batch['idepth 1']
        idep2 = sample_batch['idepth 2']
        pose1_2 = sample_batch['rela_pose']
        # euler1_2 = sample_batch['rela_euler']

        output = {}   # {0:, 1:, 'pose_pred':}
        output[0] = {}
        output[1] = {}

        if self.opt_unet.weight_map_mode:
            feature1, feature1_w = self.model_UNet(img1)
            feature2, feature2_w = self.model_UNet(img2)
            output[0]['feature_w'] = feature1_w
            output[1]['feature_w'] = feature2_w   
        else: # 2/3/4
            feature1 = self.model_UNet(img1)
            feature2 = self.model_UNet(img2)
        
        # print(torch.cuda.memory_allocated(device=self.device))

        ## process feature to SVD
        feature1_norm = None
        feature2_norm = None
        if self.opt_loss.pca_in_loss or self.opt_loss.subset_in_loss:
            feature1_3, feature1_svd = feat_svd(feature1)
            feature2_3, feature2_svd = feat_svd(feature2)
            feature1_norm = torch.norm(feature1_svd, dim=1)
            feature2_norm = torch.norm(feature2_svd, dim=1)
            feature1_norm = feature1_norm / torch.max(feature1_norm)
            feature2_norm = feature2_norm / torch.max(feature2_norm)
            if self.opt_loss.pca_in_loss:
                feature1 = feature1_svd
                feature2 = feature2_svd

            output[0]['feature_svd'] = feature1_svd
            output[1]['feature_svd'] = feature2_svd
            output[0]['feature_chnl3'] = feature1_3
            output[1]['feature_chnl3'] = feature2_3
            output[0]['feature_norm'] = feature1_norm
            output[1]['feature_norm'] = feature2_norm
        output[0]['feature'] = feature1
        output[1]['feature'] = feature2 
        

        dep1.requires_grad = False
        dep2.requires_grad = False
        pose1_2.requires_grad = False

        if self.opt_unet.pose_predict_mode:
        # with pose predictor
            euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
            pose1_2_pred = pose_from_euler_t_Tensor(euler_pred, device=self.device)
            output['pose_pred'] = pose1_2_pred
            output['euler_pred'] = euler_pred

        loss, output = self.model_loss(sample_batch, output)
        return loss, output

    def set_norm_level(self, i_batch):
        self.model_loss.set_norm_level(i_batch)
        if i_batch < 3000:
            self.model_loss.min_norm_thresh = 0
        elif i_batch < 6000:
            self.model_loss.min_norm_thresh = 0.1
        else:
            self.model_loss.min_norm_thresh = 0.2
        return


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

def gen_cam_K(source, width, height):
    '''
        CARLA and TUM have differenct definition of relation between xyz coordinate and uv coordinate.
        CARLA xyz is front-right(u)-down(v)(originally up, which is left handed, fixed to down in pose_from_euler_t function)
        TUM xyz is right(u)-down(v)-front
        Output is nparray
    '''
    assert source == 'CARLA' or source == 'TUM', 'source unrecognized'
    if source == 'CARLA':
        fx=int(width/2)
        fy=int(width/2)
        cx=int(width/2)
        cy=int(height/2)
        K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
    elif source == 'TUM':
        fx = width/640.0*525.0  # focal length x
        fy = height/480.0*525.0  # focal length y
        cx = width/640.0*319.5  # optical center x
        cy = height/480.0*239.5  # optical center y
        K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
    return K

class innerProdLoss(nn.Module):
    def __init__(self, device, opt_loss):
        super(innerProdLoss, self).__init__()
        self.device = device
        self.opt = opt_loss
        # K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ])
        # K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
        # input: x front, y right (aligned with u direction), z down (aligned with v direction) [CARLA definition]
        # output: u, v, 1
        
        self.options_from_source()

        self.K = gen_cam_K(self.opt.source, self.opt.width, self.opt.height)
        inv_K = torch.tensor(np.linalg.inv(self.K), dtype=torch.float).to(self.device)
        self.K = torch.tensor(self.K).to(self.device)

        self.uv1_grid, self.yz1_grid = gen_uvgrid(self.opt.width, self.opt.height, inv_K)
        
        self.min_norm_thresh = 0

        self.w_normalize = nn.Softmax(dim=2)

    def options_from_source(self):
        '''
        CARLA and TUM have differenct definition of relation between xyz coordinate and uv coordinate.
        CARLA xyz is front-right(u)-down(v)(originally up, which is left handed, fixed to down in pose_from_euler_t function)
        TUM xyz is right(u)-down(v)-front
        dist_coef is scaling in the exponential in the RBF kernel, related to the movement and scenario scale in the data
        '''
        assert self.opt.source == 'CARLA' or self.opt.source == 'TUM', 'source unrecognized'
        if self.opt.source == 'CARLA':
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 1.0
        elif self.opt.source == 'TUM':
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 0.1

    def set_norm_level(self, i_batch):
        # if i_batch < 500:
        #     self.dist_coef = 3e-1
        # elif i_batch < 1200:
        #     self.dist_coef = 1e-1
        # elif i_batch < 2000:
        #     self.dist_coef = 5e-2
        # else:
        #     self.dist_coef = 1e-2

        self.dist_coef = 1e-1
        # ## set the coefficient lambda of norm loss
        # if i_batch < 1000:
        #     self.norm_scale = 0
        # elif i_batch < 2000:
        #     self.norm_scale = 1e-3
        # elif i_batch < 3000:
        #     self.norm_scale = 1e-2
        # # elif i_batch < 2000:
        # #     self.norm_scale = 1
        # else:
        #     self.norm_scale = 1e-1

        # # if self.sparsify:
        # #     self.norm_scale *= 1e-2

    def calc_loss(self, flat_sel, pose1_2, pose1_2_pred=None):
        '''
        Input points are flat
        '''
        xyz1 = flat_sel[0]['xyz']
        xyz2 = flat_sel[1]['xyz']
        feature1 = flat_sel[0]['feature']
        feature2 = flat_sel[1]['feature']
        img1 = flat_sel[0]['img']
        img2 = flat_sel[1]['img']

        loss = {}

        if self.opt.opt_unet.weight_map_mode:
            feature1_w = flat_sel[0]['feature_w']
            feature2_w = flat_sel[1]['feature_w']

        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
        xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N

        if self.opt.sparsify_mode == 1:
            # 1 using a explicit weighting map, normalize L2 norm of each pixel, no norm output
            norm_mode = True
            norm_dim = 1
        elif self.opt.sparsify_mode == 2 or self.opt.sparsify_mode == 5:
            # normalize L1 norm of each channel, output L2 norm of each channel
            norm_mode = True
            norm_dim = 2
        elif self.opt.sparsify_mode == 3:
            # no normalization, output L1 norm of each channel
            norm_mode = False
            norm_dim = 2
        elif self.opt.sparsify_mode == 4:
            # no normalization, output L2 norm of each pixel
            norm_mode = False
            norm_dim = 1
        elif self.opt.sparsify_mode == 6:
            # no normalization, no norm output
            norm_mode = False
            norm_dim = 0
            
        ################# inner product of features, position and color ####################################
        gramian_feat, fea_norm_sum = gramian(feature1, feature2, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim) # could be kernelized or not
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)

        if self.opt.min_dist_mode:
            gramian_f1, _ = gramian(feature1, feature1, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim)
            pcl_diff_1 = kern_mat(xyz1, xyz1, self.dist_coef)
            gramian_f2, _ = gramian(feature2, feature2, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim)
            pcl_diff_2 = kern_mat(xyz2_trans, xyz2_trans, self.dist_coef)

        if self.opt.color_in_cost:
            gramian_color, _ = gramian(img1, img2, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim )
            if self.opt.min_dist_mode:
                gramian_c1, _ = gramian(img1, img1, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim )
                gramian_c2, _ = gramian(img2, img2, norm_mode=norm_mode, kernalize=self.opt.kernalize, norm_dim=norm_dim )
        
        # print('now original')
        # draw3DPts(xyz1, xyz2, img1, img2)
        # print('now matched')
        # draw3DPts(xyz1, xyz2_trans, img1, img2)
        ######################################################################################################

        ######################## inner product or distance of two functions ##################################
        if self.opt.diff_mode:
            if self.opt.source=='TUM':
                ### 1. perturb the scenario points by moving them to the center first
                mean_trans_2 = xyz2_trans.mean(dim=(0,2))
                pose1_2_noisy = gen_noisy_pose(self.pose1_2_noise, mean_trans_2)
                xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_trans_homo)[:, 0:3, :] # B*3*N

            elif self.opt.source=='CARLA':
                ## 2. directly perturb camera pose
                pose1_2_noisy = torch.matmul(pose1_2, self.pose1_2_noise)
                xyz2_trans_noisy = torch.matmul(pose1_2_noisy, xyz2_homo)[:, 0:3, :] # B*3*N

            pcl_diff_exp_noisy = kern_mat(xyz1, xyz2_trans_noisy, self.dist_coef)

            ### 3. treat original points (without matching) as perturbation
            # pcl_diff_exp_noisy = kern_mat(xyz1, xyz2, self.dist_coef)

            if self.opt.color_in_cost:
                if self.opt.opt_unet.weight_map_mode: ### use a weight layer
                    inner_neg_match = torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w )
                    inner_neg_noisy = torch.sum(pcl_diff_exp_noisy * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w )
                else:
                    inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat * gramian_color )
                    inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat * gramian_color )
            else:
                if self.opt.opt_unet.weight_map_mode:
                    inner_neg_match = torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                    inner_neg_noisy = torch.sum(pcl_diff_exp_noisy * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                else:
                    inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat  )
                    inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat )

            inner_neg = inner_neg_noisy / inner_neg_match

            # print('now noisy')
            # draw3DPts(xyz1, xyz2_trans_noisy, img1, img2)
        else:
            if self.opt.color_in_cost:
                if self.opt.min_dist_mode: 
                # distance between functions
                    if self.opt.opt_unet.weight_map_mode:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * gramian_c1 * feature1_w.transpose(1, 2) * feature1_w ) + \
                                    torch.sum(pcl_diff_2 * gramian_f2 * gramian_c2 * feature2_w.transpose(1, 2) * feature2_w ) - \
                                    2*torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w ) 
                    else:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * gramian_c1 ) + \
                                    torch.sum(pcl_diff_2 * gramian_f2 * gramian_c2 ) - \
                                    2*torch.sum(pcl_diff_exp * gramian_feat * gramian_color )
                else:  
                # negative inner product 
                    if self.opt.opt_unet.weight_map_mode:
                        inner_neg = - torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w ) 
                    else:
                        inner_neg = - torch.mean(pcl_diff_exp * gramian_feat * gramian_color )
            else:
                if self.opt.min_dist_mode:
                # distance between functions
                    if self.opt.opt_unet.weight_map_mode:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * feature1_w.transpose(1, 2) * feature1_w ) + \
                                    torch.sum(pcl_diff_2 * gramian_f2 * feature2_w.transpose(1, 2) * feature2_w ) - \
                                    2*torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                    else:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 ) + \
                                    torch.sum(pcl_diff_2 * gramian_f2 ) - \
                                    2*torch.sum(pcl_diff_exp * gramian_feat )
                        # a_b = torch.sum(pcl_diff_exp * gramian_feat )
                        # a_a = torch.sum(pcl_diff_1 * gramian_f1 )
                        # b_b = torch.sum(pcl_diff_2 * gramian_f2 )
                        # # print('a_b', a_b)
                        # # print('a_a', a_a)
                        # # print('b_b', b_b)
                        # inner_neg = -a_b / \
                        #     torch.sqrt(a_a * b_b )
                else:
                # negative inner product 
                    if self.opt.opt_unet.weight_map_mode:
                        inner_neg = - torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                    else:
                        inner_neg = - torch.mean(pcl_diff_exp * gramian_feat ) 

        final_loss = inner_neg
        ############################################################################################

        ############################## penalty on feature norm #####################################
        # sparsify mode 1 and 6 output norm=0
        if self.opt.sparsify_mode == 5:
            fea_norm_sum_to_loss = fea_norm_sum * 0
        else:
            fea_norm_sum_to_loss = fea_norm_sum * self.norm_scale

        if not self.opt.opt_unet.weight_map_mode:
            final_loss = final_loss + fea_norm_sum_to_loss
        #############################################################################################
        
        if self.opt.opt_unet.pose_predict_mode: # obselete
            xyz2_trans_pred = torch.matmul(pose1_2_pred, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred, self.dist_coef)
            if self.opt.color_in_cost:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat * gramian_color )
            else:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat ) 

            final_loss = final_loss + inner_neg_pred
            loss['CVO_pred'] = inner_neg_pred

        loss['final'] = final_loss
        loss['CVO'] = inner_neg
        loss['norm'] = fea_norm_sum

        return loss
        

    def forward(self, sample_batch, output): # img1 and img2 only for visualization
        
        img1 = sample_batch['image 1'] ## maybe should use image_raw
        img2 = sample_batch['image 2']
        depth1 = sample_batch['depth 1']
        depth2 = sample_batch['depth 2']
        pose1_2 = sample_batch['rela_pose']
        # img1_raw = sample_batch['image 1 raw']
        # img2_raw = sample_batch['image 2 raw']
        # idep1 = sample_batch['idepth 1']
        # idep2 = sample_batch['idepth 2']

        feature1 = output[0]['feature']
        feature2 = output[1]['feature']

        inout_flat = {}
        inout_flat[0] = {}
        inout_flat[1] = {}

        ### flatten depth and feature maps ahead of other operations to make pixel selection easier
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

        inout_flat[0]['depth'] = depth1_flat
        inout_flat[1]['depth'] = depth2_flat
        inout_flat[0]['feature'] = fea_flat_1
        inout_flat[1]['feature'] = fea_flat_2
        inout_flat[0]['img'] = img_flat_1
        inout_flat[1]['img'] = img_flat_2

        if self.opt.opt_unet.weight_map_mode:
            feature1_w = output[0]['feature_w']
            feature2_w = output[1]['feature_w']
            fea_w_flat_1 = feature1_w.reshape(batch_size, 1, n_pts_1)  # B*C*N1
            fea_w_flat_2 = feature2_w.reshape(batch_size, 1, n_pts_2)  # B*C*N2
            # fea_w_flat_1 = self.w_normalize( feature1_w.reshape(batch_size, 1, n_pts_1) ) # B*C*N1
            # fea_w_flat_2 = self.w_normalize( feature2_w.reshape(batch_size, 1, n_pts_2) ) # B*C*N2
            # fea_w_1 = fea_w_flat_1.reshape(batch_size, 1, feature1_w.shape[2], feature1_w.shape[3] )
            # fea_w_2 = fea_w_flat_2.reshape(batch_size, 1, feature2_w.shape[2], feature2_w.shape[3] )
            inout_flat[0]['feature_w'] = fea_w_flat_1
            inout_flat[1]['feature_w'] = fea_w_flat_2

        if self.opt.subset_in_loss:
            feature1_norm = output[0]['feature_norm']
            feature2_norm = output[1]['feature_norm']
            feature1_norm_flat = feature1_norm.reshape(batch_size, 1, n_pts_1)
            feature2_norm_flat = feature2_norm.reshape(batch_size, 1, n_pts_2)
            inout_flat[0]['feature_norm'] = feature1_norm_flat
            inout_flat[1]['feature_norm'] = feature2_norm_flat
            
        if self.opt.diff_mode:
            self.pose1_2_noise = gen_rand_pose(self.opt.source, self.noise_trans_scale, self.device)

        ########################## calculate loss by going over each sample in the batch ##########################
        ### Every sample should be processed separately because the selected pixels in each images are not the same.
        ### However, the calc_loss function is still designed to process a mini-batch for better compatibility. Each
        ### sample is augmented a dimension to pretend a mini-batch

        losses = {}
        losses['final'] = torch.tensor(0., device=self.device)
        losses['CVO'] = torch.tensor(0., device=self.device)
        losses['norm'] = torch.tensor(0., device=self.device)
        losses['CVO_pred'] = torch.tensor(0., device=self.device)

        if self.opt.subset_in_loss:
            mask_feat_norm_1 = [None]*int(batch_size)
            mask_feat_norm_2 = [None]*int(batch_size)

        for i in range(batch_size):
            flat_sel = {}
            flat_sel[0] = self.selected(inout_flat, 0, i) #3*N
            flat_sel[1] = self.selected(inout_flat, 1, i) #3*N  
            if self.opt.opt_unet.weight_map_mode:
                flat_sel[0]['feature_w'] = self.w_normalize(flat_sel[0]['feature_w'])
                flat_sel[1]['feature_w'] = self.w_normalize(flat_sel[1]['feature_w'])

            if self.opt.subset_in_loss:
                mask_feat_norm_1[i] = flat_sel[0]['mask_feat_norm']
                mask_feat_norm_2[i] = flat_sel[1]['mask_feat_norm']

            if not self.opt.opt_unet.pose_predict_mode:
                loss_single = self.calc_loss(flat_sel, pose1_2)
            else:
                pose1_2_pred = output['pose_pred'][i]
                loss_single = self.calc_loss(flat_sel, pose1_2, pose1_2_pred=pose1_2_pred)
                losses['CVO_pred'] += loss_single['CVO_pred']
                
            losses['final'] +=  loss_single['final']
            losses['CVO'] +=  loss_single['CVO']
            losses['norm'] +=  loss_single['norm']

        if self.opt.subset_in_loss:
            mask_feat_norm_1_stack = torch.stack(mask_feat_norm_1, dim=0).reshape(batch_size, 1, depth1.shape[2], depth1.shape[3])
            mask_feat_norm_2_stack = torch.stack(mask_feat_norm_2, dim=0).reshape(batch_size, 1, depth1.shape[2], depth1.shape[3])
            output[0]['norm_mask'] = mask_feat_norm_1_stack
            output[1]['norm_mask'] = mask_feat_norm_2_stack
            
        return losses, output

            
        #############################################
        # xyz1, xyz2 = gen_3D_flat(self.yz1_grid, depth1_flat, depth2_flat)

        # return self.calc_loss(xyz1, xyz2, pose1_2, fea_flat_1, fea_flat_2, img_flat_1, img_flat_2, pose1_2_pred)


    def selected(self, inout_flat, i, j):
        ### i is 0 or 1 (camera 0 or 1), j is index in the mini-batch
    # def selected(self, depth_flat_sample, fea_flat_sample, img_flat_sample, fea_w_flat_sample=None, fea_norm_flat_sample=None):
        '''
        Input is 1*N or 3*N
        Output 1*C*N (keep the batch dimension)
        '''
        depth_flat_sample = inout_flat[i]['depth'][j]
        fea_flat_sample = inout_flat[i]['feature'][j]
        img_flat_sample = inout_flat[i]['img'][j]

        flat_sel = {}

        ############################# create mask ############################################
        # mask = (depth_flat_sample.squeeze() > 0) & (self.uv1_grid[0] > 4) & (self.uv1_grid[0] < self.opt.width - 5) & (self.uv1_grid[1] > 4) & (self.uv1_grid[1] < self.opt.height - 5)
        mask = (depth_flat_sample.squeeze() > 0)
        
        if self.opt.subset_in_loss:
            fea_norm_flat_sample = inout_flat[i]['feature_norm'][j]
            mask_feat_norm = fea_norm_flat_sample.squeeze() >= self.min_norm_thresh
            mask = mask & mask_feat_norm
            flat_sel['mask_feat_norm'] = mask_feat_norm.unsqueeze(0)

         ##############################################################################


        depth_flat_sample_selected = depth_flat_sample[:,mask]
        fea_flat_sample_selected = fea_flat_sample[:,mask]
        img_flat_sample_selected = img_flat_sample[:,mask]
        flat_sel['depth'] = depth_flat_sample_selected.unsqueeze(0)
        flat_sel['feature'] = fea_flat_sample_selected.unsqueeze(0)
        flat_sel['img'] = img_flat_sample_selected.unsqueeze(0)

        yz1_grid_selected = self.yz1_grid[:,mask]
        xyz_selected = yz1_grid_selected * depth_flat_sample_selected
        flat_sel['xyz'] = xyz_selected.unsqueeze(0)

        if self.opt.opt_unet.weight_map_mode:
            fea_w_flat_sample = inout_flat[i]['feature_w'][j]
            fea_w_flat_sample_selected = fea_w_flat_sample[:,mask]
            flat_sel['feature_w'] = fea_w_flat_sample_selected.unsqueeze(0)

        return flat_sel