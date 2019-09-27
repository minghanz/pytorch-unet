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
        sparsify_mode=1,
        color_in_cost=False, 
        L2_norm=True, 
        pose_predict_mode=False, 
        source='TUM', 
        weight_map_mode=False, 
        min_dist_mode=True, 
        pretrained_mode=False, 
        pca_in_loss=False, 
        subset_in_loss=False
    ):
        super(UNetInnerProd, self).__init__()
        self.device = device

        assert sparsify_mode == 1 or sparsify_mode == 2 or sparsify_mode == 3 or sparsify_mode == 4 or sparsify_mode == 5 or sparsify_mode == 6, "unrecognized sparsity mode!"
        self.sparsify_mode = sparsify_mode
        self.weight_map_mode = weight_map_mode
        self.min_dist_mode = min_dist_mode
        self.pca_in_loss = pca_in_loss
        self.subset_in_loss = subset_in_loss

        if pretrained_mode:
            def final_act(x):
                return 0.05 * x
            def sigmoid_scaled(x):
                sig = nn.Sigmoid()
                x = 2 * sig(x*0.4) - 1
                return x
            if self.weight_map_mode:
                self.model_UNet = smp.UnetWithLastOut('resnet34', encoder_weights='imagenet', classes=n_classes, activation=sigmoid_scaled ).to(device)
            else: # 2/3/4
                self.model_UNet = smp.Unet('resnet34', encoder_weights='imagenet', classes=n_classes, activation=final_act ).to(device)
            for param in self.model_UNet.encoder.parameters():
                param.requires_grad = False 
            # pretrained_model = models.resnet34(pretrained=True)
            # self.model_Unet = models.unet.DynamicUnet(encoder=pretrained_model, n_classes=n_classes).to(device)
        else:
            self.model_UNet = UNet(in_channels, n_classes, depth, wf, padding, batch_norm, up_mode).to(device)

        self.model_loss = innerProdLoss(device, width, height, diff_mode, kernalize, sparsify_mode, weight_map_mode, min_dist_mode, \
            color_in_cost, L2_norm, source).to(device)
        self.pose_predict_mode = pose_predict_mode
        if self.pose_predict_mode:
            self.pose_predictor = UNetRegressor(batch_norm=True, feature_channels=n_classes*2).to(device)

    def forward(self, img1, img2, dep1, dep2, idep1, idep2, pose1_2, img1_raw, img2_raw):
        if self.weight_map_mode:
            feature1, feature1_w = self.model_UNet(img1)
            feature2, feature2_w = self.model_UNet(img2)
        else: # 2/3/4
            feature1 = self.model_UNet(img1)
            feature2 = self.model_UNet(img2)
        
        # print(torch.cuda.memory_allocated(device=self.device))

        ## process feature to SVD
        feature1_norm = None
        feature2_norm = None
        if self.pca_in_loss or self.subset_in_loss:
            feature1_3, feature1_svd = feat_svd(feature1)
            feature2_3, feature2_svd = feat_svd(feature2)
            feature1_norm = torch.norm(feature1_svd, dim=1)
            feature2_norm = torch.norm(feature2_svd, dim=1)
            feature1_norm = feature1_norm / torch.max(feature1_norm)
            feature2_norm = feature2_norm / torch.max(feature2_norm)
            if self.pca_in_loss:
                feature1 = feature1_svd
                feature2 = feature2_svd
        

        dep1.requires_grad = False
        dep2.requires_grad = False
        pose1_2.requires_grad = False

        if self.pose_predict_mode:
            # with pose predictor
            euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
            pose1_2_pred = pose_from_euler_t_Tensor(euler_pred, device=self.device)

            loss, innerp_loss, feat_norm, innerp_loss_pred = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2, pose1_2_pred=pose1_2_pred)
            return feature1, feature2, loss, innerp_loss, feat_norm, innerp_loss_pred, euler_pred
        else:
            # without pose predictor
            if self.weight_map_mode:
                loss, innerp_loss, feat_norm, feat_w_1, feat_w_2 = self.model_loss(feature1, feature2, 
                    dep1, dep2, pose1_2, img1, img2, feature1_w=feature1_w, feature2_w=feature2_w)
                return feature1, feature2, loss, innerp_loss, feat_norm, feat_w_1, feat_w_2 
            else:
                if self.pca_in_loss or self.subset_in_loss:
                    loss, innerp_loss, feat_norm, mask_norm_1, mask_norm_2 = self.model_loss(feature1, feature2, 
                        dep1, dep2, pose1_2, img1_raw, img2_raw, feature1_norm=feature1_norm, feature2_norm=feature2_norm )
                    return feature1_norm, feature2_norm, loss, innerp_loss, feat_norm, feature1_3, feature2_3, mask_norm_1, mask_norm_2
                else:
                    loss, innerp_loss, feat_norm = self.model_loss(feature1, feature2, 
                        dep1, dep2, pose1_2, img1, img2, feature1_norm=feature1_norm, feature2_norm=feature2_norm )
                    return feature1, feature2, loss, innerp_loss, feat_norm
            # loss, innerp_loss, feat_norm = self.model_loss(feature1, feature2, dep1, dep2, pose1_2, img1, img2, feature1_w=feature1_w, feature2_w=feature2_w)
            # return feature1, feature2, loss, innerp_loss, feat_norm
            # return feature1, feature2, loss, innerp_loss, feat_norm, feat_w_1, feat_w_2 

    def set_norm_level(self, i_batch):
        self.model_loss.set_norm_level(i_batch)
        if i_batch < 3000:
            self.model_loss.min_norm_thresh = 0
        elif i_batch < 6000:
            self.model_loss.min_norm_thresh = 0.1
        else:
            self.model_loss.min_norm_thresh = 0.2
        return


class innerProdLoss(nn.Module):
    def __init__(self, device, width=96, height=72, diff_mode=True, kernalize=False, sparsify_mode=False, weight_map_mode=False, min_dist_mode=True, 
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
        self.width = width
        self.height = height
        self.uv1_grid = torch.stack( (uu_grid.to(self.device), vv_grid.to(self.device), torch.ones(uu_grid.size()).to(self.device) ), dim=0 ) # 3*N, correspond to u,v,1
        self.yz1_grid = torch.mm(inv_K, self.uv1_grid).to(self.device) # 3*N, corresponding to x,y,z
        self.diff_mode = diff_mode
        self.kernalize = kernalize
        self.color_in_cost = color_in_cost
        self.L2_norm = L2_norm
        self.sparsify_mode = sparsify_mode
        self.weight_map_mode = weight_map_mode
        self.min_dist_mode = min_dist_mode
        self.min_norm_thresh = 0

        self.w_normalize = nn.Softmax(dim=2)

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
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 1.0
        elif self.source == 'TUM':
            fx = width/640.0*525.0  # focal length x
            fy = height/480.0*525.0  # focal length y
            cx = width/640.0*319.5  # optical center x
            cy = height/480.0*239.5  # optical center y
            self.K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 0.1

    def gen_rand_pose(self):
        # trans_noise = np.random.normal(scale=self.noise_trans_scale, size=(3,))
        # rot_noise = np.random.normal(scale=3.0, size=(3,))
        trans_noise = torch.randn((3), dtype=torch.float, device=self.device) * self.noise_trans_scale
        # rot_noise = torch.randn((3), dtype=torch.float, device=self.device) * 3.0
        rot_noise = torch.zeros(3, dtype=torch.float, device=self.device)
        
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

    def calc_loss(self, xyz1, xyz2, pose1_2, feature1, feature2, img1, img2, feature1_w=None, feature2_w=None, pose1_2_pred=None):
        '''
        Input points are flat
        '''
        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
        xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N

        # if self.sparsify:
        #     norm_mode = 0
        # else:
        if self.sparsify_mode == 1:
            # 1 using a explicit weighting map, normalize L2 norm of each pixel, no norm output
            norm_mode = True
            norm_dim = 1
        elif self.sparsify_mode == 2 or self.sparsify_mode == 5:
            # normalize L1 norm of each channel, output L2 norm of each channel
            norm_mode = True
            norm_dim = 2
        elif self.sparsify_mode == 3:
            # no normalization, output L1 norm of each channel
            norm_mode = False
            norm_dim = 2
        elif self.sparsify_mode == 4:
            # no normalization, output L2 norm of each pixel
            norm_mode = False
            norm_dim = 1
        elif self.sparsify_mode == 6:
            # no normalization, no norm output
            norm_mode = False
            norm_dim = 0
            
        # gramian_feat, fea_norm_sum = gramian(feature1, feature2, norm_mode=norm_mode, kernalize=self.kernalize, L2_norm=self.L2_norm) # could be kernelized or not
        gramian_feat, fea_norm_sum = gramian(feature1, feature2, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim) # could be kernelized or not
        pcl_diff_exp = kern_mat(xyz1, xyz2_trans, self.dist_coef)

        if self.min_dist_mode:
            gramian_f1, _ = gramian(feature1, feature1, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim)
            pcl_diff_1 = kern_mat(xyz1, xyz1, self.dist_coef)
            gramian_f2, _ = gramian(feature2, feature2, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim)
            pcl_diff_2 = kern_mat(xyz2_trans, xyz2_trans, self.dist_coef)

        if self.color_in_cost:
            gramian_color, _ = gramian(img1, img2, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim )
            if self.min_dist_mode:
                gramian_c1, _ = gramian(img1, img1, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim )
                gramian_c2, _ = gramian(img2, img2, norm_mode=norm_mode, kernalize=self.kernalize, norm_dim=norm_dim )
                
                # print(img1)
        
        # print('now original')
        # draw3DPts(xyz1, xyz2, img1, img2)
        # print('now matched')
        # draw3DPts(xyz1, xyz2_trans, img1, img2)

        # sparsify mode 1 and 6 output norm=0
        if self.sparsify_mode == 5:
            fea_norm_sum_to_loss = fea_norm_sum * 0
        else:
            fea_norm_sum_to_loss = fea_norm_sum * self.norm_scale

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
                if self.sparsify_mode != 1:
                    # inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat * gramian_color ) #  * gramian_feat
                    inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat * gramian_color )
                    inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat * gramian_color )
                else:
                    inner_neg_match = torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w )
                    inner_neg_noisy = torch.sum(pcl_diff_exp_noisy * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w )
                inner_neg = inner_neg_noisy / inner_neg_match
            else:
                if self.sparsify_mode != 1:
                    # inner_neg = - torch.sum(pcl_diff_exp_diff * gramian_feat ) #  * gramian_feat
                    inner_neg_match = torch.mean(pcl_diff_exp * gramian_feat  )
                    inner_neg_noisy = torch.mean(pcl_diff_exp_noisy * gramian_feat )
                else:
                    inner_neg_match = torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                    inner_neg_noisy = torch.sum(pcl_diff_exp_noisy * gramian_feat * feature1_w.transpose(1, 2) * feature2_w )
                inner_neg = inner_neg_noisy / inner_neg_match
                # print(inner_neg_match)

            final_loss = inner_neg
            if self.sparsify_mode != 1:
                final_loss = final_loss + fea_norm_sum_to_loss

            # print('inner_neg', inner_neg)
            # print('now noisy')
            # draw3DPts(xyz1, xyz2_trans_noisy, img1, img2)

        else:
            if self.color_in_cost:
                if self.min_dist_mode:
                    if not self.weight_map_mode:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * gramian_c1 ) + \
                            torch.sum(pcl_diff_2 * gramian_f2 * gramian_c2 ) - 2 * torch.sum(pcl_diff_exp * gramian_feat * gramian_color )
                    else:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * gramian_c1 * feature1_w.transpose(1, 2) * feature1_w ) + \
                            torch.sum(pcl_diff_2 * gramian_f2 * gramian_c2 * feature2_w.transpose(1, 2) * feature2_w ) - \
                            2 *torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w ) 
                else:
                    if not self.weight_map_mode:
                        inner_neg = - torch.mean(pcl_diff_exp * gramian_feat * gramian_color ) # * gramian_feat
                    else:
                        inner_neg = - torch.sum(pcl_diff_exp * gramian_feat * gramian_color * feature1_w.transpose(1, 2) * feature2_w ) # * gramian_feat
            else:
                if self.min_dist_mode:
                    if not self.weight_map_mode:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 ) + \
                            torch.sum(pcl_diff_2 * gramian_f2 ) - 2 * torch.sum(pcl_diff_exp * gramian_feat )
                        # print('no color, min_dist, no mask')

                        # a_b = torch.sum(pcl_diff_exp * gramian_feat )
                        # a_a = torch.sum(pcl_diff_1 * gramian_f1 )
                        # b_b = torch.sum(pcl_diff_2 * gramian_f2 )
                        # # print('a_b', a_b)
                        # # print('a_a', a_a)
                        # # print('b_b', b_b)
                        # inner_neg = -a_b / \
                        #     torch.sqrt(a_a * b_b )

                        # print('inner_neg', inner_neg)
                    else:
                        inner_neg = torch.sum(pcl_diff_1 * gramian_f1 * feature1_w.transpose(1, 2) * feature1_w ) + \
                        torch.sum(pcl_diff_2 * gramian_f2 * feature2_w.transpose(1, 2) * feature2_w ) - \
                        torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w ) 
                    
                elif not self.weight_map_mode:
                    inner_neg = - torch.mean(pcl_diff_exp * gramian_feat ) # * gramian_feat
                else:
                    inner_neg = - torch.sum(pcl_diff_exp * gramian_feat * feature1_w.transpose(1, 2) * feature2_w ) # * gramian_feat

            final_loss = inner_neg
            if not self.weight_map_mode:
                final_loss = final_loss + fea_norm_sum_to_loss
        
        if pose1_2_pred is not None:
            xyz2_trans_pred = torch.matmul(pose1_2_pred, xyz2_homo)[:, 0:3, :] # B*3*N
            pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred, self.dist_coef)
            if self.color_in_cost:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat * gramian_color )
            else:
                inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat ) 

            final_loss = final_loss + inner_neg_pred

            
            return final_loss, inner_neg, fea_norm_sum, inner_neg_pred
            # return final_loss, inner_neg, inner_neg_pred

        return final_loss, inner_neg, fea_norm_sum
        # return final_loss, inner_neg


    def forward(self, feature1, feature2, depth1, depth2, pose1_2, img1, img2, feature1_w=None, feature2_w=None, pose1_2_pred=None, feature1_norm=None, feature2_norm=None): # img1 and img2 only for visualization
        
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

        if self.weight_map_mode:
            fea_w_flat_1 = feature1_w.reshape(batch_size, 1, n_pts_1)  # B*C*N1
            fea_w_flat_2 = feature2_w.reshape(batch_size, 1, n_pts_2)  # B*C*N2
            # fea_w_flat_1 = self.w_normalize( feature1_w.reshape(batch_size, 1, n_pts_1) ) # B*C*N1
            # fea_w_flat_2 = self.w_normalize( feature2_w.reshape(batch_size, 1, n_pts_2) ) # B*C*N2
            # fea_w_1 = fea_w_flat_1.reshape(batch_size, 1, feature1_w.shape[2], feature1_w.shape[3] )
            # fea_w_2 = fea_w_flat_2.reshape(batch_size, 1, feature2_w.shape[2], feature2_w.shape[3] )

        if feature1_norm is not None and feature2_norm is not None:
            feature1_norm_flat = feature1_norm.reshape(batch_size, 1, n_pts_1)
            feature2_norm_flat = feature2_norm.reshape(batch_size, 1, n_pts_2)
        else:
            feature1_norm_flat = [None]*int(batch_size)
            feature2_norm_flat = [None]*int(batch_size)
            


        #################################################
        final_loss = torch.tensor(0., device=self.device)
        inner_neg = torch.tensor(0., device=self.device)
        fea_norm_sum = torch.tensor(0., device=self.device)
        inner_neg_pred = torch.tensor(0., device=self.device)

        mask_feat_norm_1 = [None]*int(batch_size)
        mask_feat_norm_2 = [None]*int(batch_size)

        for i in range(batch_size):
            if self.weight_map_mode:
                xyz1, dep_flat_1_sel, fea_flat_1_sel, img_flat_1_sel, fea_w_flat_1_sel = self.selected(depth1_flat[i], 
                                                                    fea_flat_1[i], img_flat_1[i], fea_w_flat_1[i]) #3*N
                xyz2, dep_flat_2_sel, fea_flat_2_sel, img_flat_2_sel, fea_w_flat_2_sel = self.selected(depth2_flat[i], 
                                                                    fea_flat_2[i], img_flat_2[i], fea_w_flat_2[i]) #3*N   
                fea_w_flat_1_sel = self.w_normalize(fea_w_flat_1_sel)
                fea_w_flat_2_sel = self.w_normalize(fea_w_flat_2_sel)
                
            else:
                xyz1, dep_flat_1_sel, fea_flat_1_sel, img_flat_1_sel, mask_feat_norm_1[i] = self.selected(depth1_flat[i], fea_flat_1[i], img_flat_1[i], fea_norm_flat_sample=feature1_norm_flat[i]) #3*N
                xyz2, dep_flat_2_sel, fea_flat_2_sel, img_flat_2_sel, mask_feat_norm_2[i] = self.selected(depth2_flat[i], fea_flat_2[i], img_flat_2[i], fea_norm_flat_sample=feature2_norm_flat[i]) #3*N 
                fea_w_flat_1_sel = None
                fea_w_flat_2_sel = None

            if pose1_2_pred is None:
                final_loss_single, inner_neg_single, fea_norm_sum_single = \
                    self.calc_loss(xyz1, xyz2, pose1_2[i], fea_flat_1_sel, fea_flat_2_sel, img_flat_1_sel, img_flat_2_sel,
                        feature1_w=fea_w_flat_1_sel, feature2_w=fea_w_flat_2_sel)
            else:
                final_loss_single, inner_neg_single, fea_norm_sum_single, inner_neg_pred_single = \
                    self.calc_loss(xyz1, xyz2, pose1_2[i], fea_flat_1_sel, fea_flat_2_sel, img_flat_1_sel, img_flat_2_sel, 
                        feature1_w=fea_w_flat_1_sel, feature2_w=fea_w_flat_2_sel, pose1_2_pred=pose1_2_pred[i])

                inner_neg_pred = inner_neg_pred + inner_neg_pred_single

            final_loss = final_loss + final_loss_single
            inner_neg = inner_neg + inner_neg_single
            fea_norm_sum = fea_norm_sum + fea_norm_sum_single

        if feature1_norm is not None and feature2_norm is not None:
            mask_feat_norm_1_stack = torch.stack(mask_feat_norm_1, dim=0).reshape(batch_size, 1, depth1.shape[2], depth1.shape[3])
            mask_feat_norm_2_stack = torch.stack(mask_feat_norm_2, dim=0).reshape(batch_size, 1, depth1.shape[2], depth1.shape[3])
            

        if pose1_2_pred is None:
            if self.weight_map_mode:
                return final_loss, inner_neg, fea_norm_sum, feature1_w, feature2_w
                # return final_loss, inner_neg, fea_w_1, fea_w_2
            elif feature1_norm is not None and feature2_norm is not None:
                return final_loss, inner_neg, fea_norm_sum, mask_feat_norm_1_stack, mask_feat_norm_2_stack
            else:
                return final_loss, inner_neg, fea_norm_sum
                # return final_loss, inner_neg
                
        else:
            return final_loss, inner_neg, fea_norm_sum, inner_neg_pred
            # return final_loss, inner_neg, inner_neg_pred

            
        #############################################
        # xyz1, xyz2 = gen_3D_flat(self.yz1_grid, depth1_flat, depth2_flat)

        # return self.calc_loss(xyz1, xyz2, pose1_2, fea_flat_1, fea_flat_2, img_flat_1, img_flat_2, pose1_2_pred)


    def selected(self, depth_flat_sample, fea_flat_sample, img_flat_sample, fea_w_flat_sample=None, fea_norm_flat_sample=None):
        '''
        Input is 1*N or 3*N
        Output 1*C*N (keep the batch dimension)
        '''

        # feat_flat_svded = feat_svd_flat_single(fea_flat_sample)
        # feat_norm = torch.norm(feat_flat_svded, dim=0)
        # mask_feat_norm = feat_norm >= 0.2 * torch.max(feat_norm)
        # print(torch.sum(mask_feat_norm.to(torch.float32)))

        # mask = (depth_flat_sample.squeeze() > 0) & (self.uv1_grid[0] > 4) & (self.uv1_grid[0] < self.width - 5) & (self.uv1_grid[1] > 4) & (self.uv1_grid[1] < self.height - 5)
        mask = (depth_flat_sample.squeeze() > 0)
        
        if fea_norm_flat_sample is not None:
            mask_feat_norm = fea_norm_flat_sample.squeeze() >= self.min_norm_thresh
            mask = mask & mask_feat_norm
            mask_feat_norm = mask_feat_norm.unsqueeze(0)
        else:
            mask_feat_norm = None

        depth_flat_sample_selected = depth_flat_sample[:,mask]
        fea_flat_sample_selected = fea_flat_sample[:,mask]
        img_flat_sample_selected = img_flat_sample[:,mask]
        if fea_w_flat_sample is not None:
            fea_w_flat_sample_selected = fea_w_flat_sample[:,mask]

        yz1_grid_selected = self.yz1_grid[:,mask]
        xyz_selected = yz1_grid_selected * depth_flat_sample_selected

        if fea_w_flat_sample is not None:
            return xyz_selected.unsqueeze(0), depth_flat_sample_selected.unsqueeze(0), \
                fea_flat_sample_selected.unsqueeze(0), img_flat_sample_selected.unsqueeze(0), fea_w_flat_sample_selected.unsqueeze(0)
        else:
            return xyz_selected.unsqueeze(0), depth_flat_sample_selected.unsqueeze(0), \
                fea_flat_sample_selected.unsqueeze(0), img_flat_sample_selected.unsqueeze(0), mask_feat_norm