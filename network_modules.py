import torch
import torch.nn as nn
from unet import UNet
from unet_pose_regressor import UNetRegressor
import numpy as np
from dataloader import pose_from_euler_t, pose_from_euler_t_Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from geometry_plot import draw3DPts

from geometry import kern_mat, gramian, gen_3D, gen_3D_flat, rgb_to_hsv, gen_rand_pose, gen_noisy_pose, gen_uvgrid #, gen_cam_K

from log import remove_edge
from gaussian_smooth import GaussianSmoothing
import torch.nn.functional as F

# from fastai.vision import *
# from torchvision import models
import segmentation_models_pytorch as smp

# from run import feat_pca

def feat_pca(feature):
    # input feature is b*c*h*w, need to change to b*c*n
    b = feature.shape[0]
    c = feature.shape[1]
    h = feature.shape[2]
    w = feature.shape[3]
    feat_flat = feature.reshape(b, c, h*w)

    feat_new = feat_pca_flat(feat_flat)

    feat_img = feat_new.reshape(b,c,h,w)
    feat_img_3 = feat_img[:,0:3,:,:]
    return feat_img_3, feat_img

def feat_pca_flat(feat_flat):
    u, s, v = torch.svd(feat_flat)
    # feat_new = torch.bmm( u[:,:,0:3].transpose(1,2), feat_flat) # b*3*n
    # feat_img = feat_new.reshape(b,3,h,w)
    feat_new = torch.bmm( u.transpose(1,2), feat_flat) # b*3*n
    return feat_new

def feat_pca_flat_single(feat_flat):
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

    def forward(self, inputs, i_batch=None):
        
        if i_batch is not None:
            self.iteration_related_setting(i_batch)

        outputs = self.feature_map_generation(inputs)

        # dep1.requires_grad = False
        # dep2.requires_grad = False
        # pose1_2.requires_grad = False

        # if self.opt_unet.pose_predict_mode:
        # # with pose predictor
        #     euler_pred = self.pose_predictor(feature1, feature2, idep1, idep2)
        #     pose1_2_pred = pose_from_euler_t_Tensor(euler_pred, device=self.device)
        #     outputs['pose_pred'] = pose1_2_pred
        #     outputs['euler_pred'] = euler_pred

        loss, outputs = self.model_loss(inputs, outputs, i_batch)
        return loss, outputs

    def feature_map_generation(self, inputs):
        outputs = {}   # {0:, 1:, 'pose_pred':}
        outputs[0] = {}
        outputs[1] = {}

        ####### Generate feature map from UNet
        for i in range(2):
            ## load input
            img = inputs[i]['img']

            ## run through unet
            if self.opt_unet.weight_map_mode:
                feature, feature_w = self.model_UNet(img)
                outputs[i]['feature_w'] = feature_w 
            else:
                feature = self.model_UNet(img)

            # if self.opt_loss.kernalize:
            #     ## centralize feature
            #     feature_centered = feature - torch.mean(feature, dim=(2,3), keepdim=True)
            #     # ## manually zeroing edge part
            #     if self.opt_loss.zero_edge_region:
            #         remove_edge(feature_centered)
            # else:
            #     feature_centered = feature
            # ## normalize the max of feature norm of any pixel to 1
            # feature_norm = torch.norm(feature_centered, dim=1, keepdim=True)
            # feature_norm = feature_norm / torch.max(feature_norm)
            # outputs[i]['feature_norm'] = feature_norm
            # ## PCA
            # if self.opt_loss.pca_in_loss or self.opt_loss.visualize_pca_chnl:
            #     feature_3, feature_pca = feat_pca(feature_centered)
            #     outputs[i]['feature_pca'] = feature_pca
            #     outputs[i]['feature_chnl3'] = feature_3
            # ## what is used in calculation of loss
            # if self.opt_loss.pca_in_loss:
            #     outputs[i]['feature'] = feature_pca
            # else:
            #     outputs[i]['feature'] = feature_centered # feature or feature_centered? 

            outputs[i]['feature'] = feature
            outputs[i]['feature_normalized'] = torch.zeros_like(outputs[i]['feature'])
            
        return outputs


    def iteration_related_setting(self, i_batch):
        self.model_loss.set_norm_level(i_batch)
        return

class innerProdLoss(nn.Module):
    def __init__(self, device, opt_loss):
        super(innerProdLoss, self).__init__()
        ## Device and options
        self.device = device
        self.opt = opt_loss
        self.options_from_source()

        # ## Intrinsic matrix
        # self.K = gen_cam_K(self.opt.source, self.opt.width, self.opt.height)
        # inv_K = torch.tensor(np.linalg.inv(self.K), dtype=torch.float).to(self.device)
        # self.K = torch.tensor(self.K).to(self.device)

        # ## For generating point cloud
        # self.uv1_grid, self.yz1_grid = gen_uvgrid(self.opt.width, self.opt.height, inv_K)
        self.gen_cam_K_and_grid()
        
        ## For selecting pixels according to the norm of the feature vector, when self.opt.subset_in_loss is True
        self.min_norm_thresh = 0

        ## For normalizing the weight map to [0,1], when self.opt.opt_unet.weight_map_mode is True
        self.w_normalize = nn.Softmax(dim=2)

        ## For calculating gradient of grayscale image
        self.u_pad = torch.nn.ReflectionPad2d((1,1,0,0))
        self.v_pad = torch.nn.ReflectionPad2d((0,0,1,1))
        self.u_grad_kern = torch.tensor([[-0.5, 0, 0.5]], device=self.device).reshape(1,1,1,3)
        self.v_grad_kern = torch.tensor([[-0.5], [0], [0.5]], device=self.device).reshape(1,1,3,1)

        ## For setting the initial dist_coef (scale)
        self.dist_coef = self.opt.dist_coef
        
    def gen_cam_K_and_grid(self):
        '''
            CARLA and TUM have differenct definition of relation between xyz coordinate and uv coordinate.
            CARLA xyz is front-right(u)-down(v)(originally up, which is left handed, fixed to down in pose_from_euler_t function)
            TUM xyz is right(u)-down(v)-front
            Output is nparray
        '''
        assert self.opt.source == 'CARLA' or self.opt.source == 'TUM', 'source unrecognized'
        if self.opt.source == 'CARLA':
            fx=int(self.opt.width/2)
            fy=int(self.opt.width/2)
            cx=int(self.opt.width/2)
            cy=int(self.opt.height/2)

            K = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
            inv_K = torch.tensor(np.linalg.inv(K), dtype=torch.float).to(self.device)
            self.uv1_grid, self.yz1_grid = gen_uvgrid(self.opt.width, self.opt.height, inv_K)

        elif self.opt.source == 'TUM':
            #### see: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
            fx = self.opt.width/640.0*525.0  # focal length x
            fy = self.opt.height/480.0*525.0  # focal length y
            cx = self.opt.width/640.0*319.5  # optical center x
            cy = self.opt.height/480.0*239.5  # optical center y

            if self.opt.keep_scale_consistent and (not (self.opt.run_eval or self.opt.trial_mode) ):
                num_width = self.opt.width_split
                num_height = self.opt.height_split
                new_h = self.opt.effective_height
                new_w = self.opt.effective_width
                K = {}
                inv_K = {}
                self.uv1_grid_dict = {}
                self.yz1_grid_dict = {}

                K_ = np.array([ [cx, fx, 0], [cy, 0, fy], [1, 0, 0] ]) 
                inv_K_ = torch.tensor(np.linalg.inv(K_), dtype=torch.float).to(self.device)
                self.uv1_grid_dict['original'], self.yz1_grid_dict['original'] = gen_uvgrid(self.opt.width, self.opt.height, inv_K_)

                for i in range(num_height):
                    for j in range(num_width):
                        start_h = i * new_h
                        start_w = j * new_w
                        cx_cur = cx - start_w
                        cy_cur = cy - start_h
                        K[(i,j)] = np.array([ [fx, 0, cx_cur], [0, fy, cy_cur], [0, 0, 1] ])
                        inv_K[(i,j)] = torch.tensor(np.linalg.inv( K[(i,j)] ), dtype=torch.float).to(self.device)
                        self.uv1_grid_dict[(i,j)], self.yz1_grid_dict[(i,j)] = gen_uvgrid(new_w, new_h, inv_K[(i,j)] )
            else:
                K = np.array([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]) 
                inv_K = torch.tensor(np.linalg.inv(K), dtype=torch.float).to(self.device)
                self.uv1_grid, self.yz1_grid = gen_uvgrid(self.opt.width, self.opt.height, inv_K)

        return 


    def options_from_source(self):
        '''
        noise_trans_scale related to the purturbation to the true pose, used to calculate the sensitivity to the pose
        '''
        assert self.opt.source == 'CARLA' or self.opt.source == 'TUM', 'source unrecognized'
        if self.opt.source == 'CARLA':
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 1.0
        elif self.opt.source == 'TUM':
            # self.dist_coef = 1e-1
            self.noise_trans_scale = 0.1

    def set_norm_level(self, i_batch):
        '''
        dist_coef is scaling in the exponential in the RBF kernel, related to the movement and scenario scale in the data
        '''
        # if i_batch < 500:
        #     self.dist_coef = 3e-1
        # elif i_batch < 1200:
        #     self.dist_coef = 1e-1
        # elif i_batch < 2000:
        #     self.dist_coef = 5e-2
        # else:
        #     self.dist_coef = 1e-2
        ## set the coefficient lambda of norm loss in total loss
        # self.norm_scale = 1e-3

        # self.dist_coef = 1e-1
        # self.dist_coef = {}
        # self.dist_coef['xyz_align'] = 0.1
        # self.dist_coef['img'] = 0.1
        # self.dist_coef['feature'] = 1 ###?
        # self.dist_coef['xyz_noisy'] = 0.1

        if i_batch < 3000:
            self.min_norm_thresh = 0
        elif i_batch < 6000:
            self.min_norm_thresh = 0.1
        else:
            self.min_norm_thresh = 0.2


    def forward(self, inputs, outputs, i_batch=None): # img1 and img2 only for visualization
        self.i_batch = i_batch
        pose1_2 = inputs['rela_pose']

        if self.opt.keep_scale_consistent and (not (self.opt.run_eval or self.opt.trial_mode) ):
            if self.opt.eval_full_size:
                self.yz1_grid = self.yz1_grid_dict['original']
            else:
                ij = ( int(inputs['ij'][0]), int(inputs['ij'][1]) )
                self.yz1_grid = self.yz1_grid_dict[ ij ]

        self.gen_mask(inputs, outputs)
        self.gen_grad_map(inputs, outputs)

        inout_flat = self.flatten_inputs_outputs(inputs, outputs)
            
        if self.opt.diff_mode:
            self.pose1_2_noise = gen_rand_pose(self.opt.source, self.noise_trans_scale, self.device)

        ########################## calculate loss by going over each sample in the batch ##########################
        ### Every sample should be processed separately because the selected pixels in each images are not the same.
        ### However, the calc_loss function is still designed to process a mini-batch for better compatibility. Each
        ### sample is augmented a dimension to pretend a mini-batch

        losses = {}
        losses['final'] = torch.tensor(0., device=self.device)
        batch_size = inputs[0]['img'].shape[0]

        valid_single_loss = False
        for i in range(batch_size):
            flat_sel = {} # a single data (pair) to be fed to masking and calculating loss 
            flat_sel[0] = self.selected(inout_flat, 0, i) #3*N
            flat_sel[1] = self.selected(inout_flat, 1, i) #3*N  

            if flat_sel[0]['xyz'].shape[-1] == 0 or flat_sel[1]['xyz'].shape[-1] == 0:
                continue

            ### Post process after selection
            self.normalize_feature_after_sel(flat_sel)
            ## Put the feature_normalized in fea_flat to output
            self.output_feature_normalized(flat_sel, outputs, i)

            if not self.opt.no_inner_prod:
                if not self.opt.opt_unet.pose_predict_mode:
                    if self.opt.diff_mode:
                        loss_single, loss_single_noisy = self.calc_loss(flat_sel, pose1_2)
                    else:
                        loss_single = self.calc_loss(flat_sel, pose1_2)
                # else:
                #     pose1_2_pred = outputs['pose_pred'][i]
                #     loss_single = self.calc_loss(flat_sel, pose1_2, pose1_2_pred=pose1_2_pred)
                #     losses['CVO_pred'] += loss_single['CVO_pred']    

                for item in loss_single:
                    if i == 0:
                        losses[item] = torch.tensor(0., device=self.device)
                    losses[item] += loss_single[item]
                valid_single_loss = True
        
        if valid_single_loss:
            for i in range(len(self.opt.loss_item)):
                losses['final'] += losses[self.opt.loss_item[i]] * self.opt.loss_weight[i]
        
        self.output_feature_norm(outputs, losses)
            
        return losses, outputs

    def gen_grad_map(self, inputs, outputs):

        for i in range(2):
            outputs[i]['grad_u'] = self.grad_img(inputs[i]['gray'], direction='u') + 0.5 # make it range [0,1]
            outputs[i]['grad_v'] = self.grad_img(inputs[i]['gray'], direction='v') + 0.5
        
        return

    def grad_img(self, gray, direction):
        if direction == 'u':
            gray_pad = self.u_pad(gray)
            grad = nn.functional.conv2d(gray_pad, self.u_grad_kern)
        elif direction == 'v':
            gray_pad = self.v_pad(gray)
            grad = nn.functional.conv2d(gray_pad, self.v_grad_kern)
        else:
            raise ValueError("Unrecognized direction!")

        return grad

    def gen_mask(self, inputs, outputs):
        for i in range(2):
            outputs[i]['mask'] = inputs[i]['depth'] > 0
            if self.opt.zero_edge_region:
                remove_edge(outputs[i]['mask'], byte_mode=True)
            # mask = (depth_flat_sample.squeeze() > 0) & (self.uv1_grid[0] > 4) & (self.uv1_grid[0] < self.opt.width - 5) & (self.uv1_grid[1] > 4) & (self.uv1_grid[1] < self.opt.height - 5)
            if self.opt.subset_in_loss:
                outputs[i]['norm_mask'] = outputs[i]['feature_norm'] >= self.min_norm_thresh
                outputs[i]['mask'] = outputs[i]['norm_mask'] & outputs[i]['mask']

        return

    def flatten_inputs_outputs(self, inputs, outputs):
        inout_flat = {}
        inout_flat[0] = {}
        inout_flat[1] = {}

        items_to_be_reshaped = ['depth', 'img', 'feature', 'grad_u', 'grad_v', 'mask']
        if self.opt.opt_unet.weight_map_mode:
            items_to_be_reshaped.append('feature_w')

        for i in range(2):
            for item in items_to_be_reshaped:
                if item in inputs[i]:
                    inout_flat[i][item] = inputs[i][item].reshape(inputs[i][item].shape[0], inputs[i][item].shape[1], -1)
                elif item in outputs[i]:
                    inout_flat[i][item] = outputs[i][item].reshape(outputs[i][item].shape[0], outputs[i][item].shape[1], -1)
                else:
                    print("Inputs: ", list(inputs[i].keys()) )
                    print("Outputs: ", list(outputs[i].keys()) )
                    raise ValueError("The item {} to be reshaped is not in either inputs or outputs! ".format(item))
                # fea_w_flat_1 = self.w_normalize( feature1_w.reshape(batch_size, 1, n_pts_1) ) # B*C*N1
                # fea_w_1 = fea_w_flat_1.reshape(batch_size, 1, feature1_w.shape[2], feature1_w.shape[3] )
        return inout_flat

    
    def selected(self, inout_flat, i, j):
        ### i is 0 or 1 (camera 0 or 1), j is index in the mini-batch
    # def selected(self, depth_flat_sample, fea_flat_sample, img_flat_sample, fea_w_flat_sample=None, fea_norm_flat_sample=None):
        '''
        Input is 1*N or 3*N
        Output 1*C*N (keep the batch dimension)
        '''
        mask = inout_flat[i]['mask'][j].squeeze()
        
        flat_sel = {}

        items_to_select = ['depth', 'feature', 'img', 'grad_u', 'grad_v']
        if self.opt.opt_unet.weight_map_mode:
            items_to_select.append('feature_w')

        for item in items_to_select:
            flat_sel[item] = inout_flat[i][item][j][:,mask].unsqueeze(0) # pretend to be a mini-batch of batch size 1

        yz1_grid_selected = self.yz1_grid[:,mask]
        xyz_selected = yz1_grid_selected * flat_sel['depth'].squeeze(0)
        flat_sel['xyz'] = xyz_selected.unsqueeze(0)

        if self.opt.opt_unet.weight_map_mode:
            flat_sel['feature_w'] = self.w_normalize(flat_sel['feature_w'])

        return flat_sel

    def normalize_feature_after_sel(self, flat_sel):
        if self.opt.sparsify_mode == 1:
            # 1 using a explicit weighting map, normalize L2 norm of each pixel, no norm output
            self.norm_mode_for_feat_gram = True
            self.norm_dim = 1
        elif self.opt.sparsify_mode == 2:
            # normalize L1 norm of each channel, output L2 norm of each channel
            self.norm_mode_for_feat_gram = True
            self.norm_dim = 2
        elif self.opt.sparsify_mode == 5:
            # normalize L1 norm of each batch, output L2 norm of each channel
            self.norm_mode_for_feat_gram = True
            self.norm_dim = (1,2)

        elif self.opt.sparsify_mode == 3:
            # no normalization, output L1 norm of each channel
            self.norm_mode_for_feat_gram = False
            self.norm_dim = 2
        elif self.opt.sparsify_mode == 4:
            # no normalization, output L2 norm of each pixel
            self.norm_mode_for_feat_gram = False
            self.norm_dim = 1
        elif self.opt.sparsify_mode == 6:
            # no normalization, no norm output
            self.norm_mode_for_feat_gram = False
            self.norm_dim = 0
        ### Originally, use L2 norm when norm_dim == 1, use L1 norm when norm_dim == 2 or (1,2)
        ### Not defined: L2 norm when norm_dim = (1,2)

        for i in range(2):
            fea_flat = flat_sel[i]['feature']
            flat_sel[i]['feature_normalized'] = fea_flat
            flat_sel[i]['feature_norm_sum'] = torch.tensor(0., dtype=fea_flat.dtype, device=fea_flat.device )

            ## Skip if we want to no centralization and no normalization
            if self.norm_dim == 1 or self.norm_dim == 2 or self.norm_dim == (1,2):
                ## Calculate the mean and centralize
                fea_mean = torch.mean(fea_flat, dim=self.norm_dim, keepdim=True)
                fea_centered = fea_flat - fea_mean

                ## Calculate the norm along a certain dimension with the L1 or L2 norm
                if self.opt.L_norm == 1:
                    fea_norm = torch.mean(torch.abs(fea_centered), dim=self.norm_dim, keepdim=True) # L1 norm across channels # B*1*N / pixels # B*C*1 / pixel and channels * B*1*1
                elif self.opt.L_norm == 2:
                    fea_norm = torch.norm(fea_centered, dim=self.norm_dim, keepdim=True) #L2 norm across channels # B*1*N

                flat_sel[i]['feature_mean'] = fea_mean
                flat_sel[i]['feature_norm'] = fea_norm

                # print("fea_mean", fea_mean)
                # print("fea_norm", fea_norm)

                fea_norm_sum = torch.mean(fea_norm)

                ## if self.norm_mode_for_feat_gram is False, feature_normalized is feature
                if self.norm_mode_for_feat_gram == True:
                    flat_sel[i]['feature_normalized'] = torch.div(fea_centered, fea_norm )

                    if self.norm_dim == 2 or self.norm_dim == (1,2):
                        flat_sel[i]['feature_normalized'] *= self.opt.feat_scale_after_normalize
                        fea_norm_sum = torch.mean( torch.norm(flat_sel[i]['feature_normalized'], dim=2) )
                    else:
                        flat_sel[i]['feature_normalized'] *= self.opt.feat_norm_per_pxl
                        ### norm_dim== 1 (reduce the channel dimension) cannot be consistent with output map (the norm is in each pixel)

                flat_sel[i]['feature_norm_sum'] = fea_norm_sum

        return
    
    ## Normalize the overall feature map using the statistics of the selected pixels
    def output_feature_normalized(self, flat_sel, outputs, idx_in_batch):
        for i in range(2):
            if self.norm_mode_for_feat_gram:
                fea_norm = flat_sel[i]['feature_norm']
                fea_mean = flat_sel[i]['feature_mean']

                if self.norm_dim == 2 or self.norm_dim == (1,2):
                    fea_mean = fea_mean.unsqueeze(3) ## B*C*1*1 or B*1*1*1
                    fea_norm = fea_norm.unsqueeze(3) ## B*C*1*1 or B*1*1*1
                    fea_centered = outputs[i]['feature'][idx_in_batch:idx_in_batch+1] - fea_mean
                    outputs[i]['feature_normalized'][idx_in_batch:idx_in_batch+1] = torch.div(fea_centered, fea_norm)
                    outputs[i]['feature_normalized'][idx_in_batch:idx_in_batch+1] *= self.opt.feat_scale_after_normalize

                elif self.norm_dim == 1:
                    outputs[i]['feature_normalized'][idx_in_batch:idx_in_batch+1] = outputs[i]['feature'][idx_in_batch:idx_in_batch+1]
                    ### cannot apply normalization for each pixel using a subset of pixels
            else:
                outputs[i]['feature_normalized'][idx_in_batch:idx_in_batch+1] = outputs[i]['feature'][idx_in_batch:idx_in_batch+1] ### Note: here it is not aligned with what's in normalize_feature_after_sel

        return
    
    def output_feature_norm(self, outputs, losses):
        for i in range(2):
            ### It's possible that output_feature_normalized is not run, then feature_normalized will be all zeros
            ### 10/25: validate_sample function added checking of valid depth number, so this may not be necessary now
            if torch.max(outputs[i]['feature_normalized']) == 0:
                print("nan will happen!!!!!!!!!!!!!!!")
                feature = outputs[i]['feature']
                outputs[i]['feature_normalized'] = feature
                #### centralize and normalize
                if self.norm_mode_for_feat_gram == True:
                    if self.norm_dim == 2 or self.norm_dim == (1,2):
                        if self.norm_dim == 2:
                            norm_dim = (2,3)
                        elif self.norm_dim == (1,2):
                            norm_dim = (1,2,3)
                        fea_mean = torch.mean(feature, dim=norm_dim, keepdim=True)
                        fea_centered = feature - fea_mean

                        if self.opt.L_norm == 1:
                            fea_norm = torch.mean(torch.abs(fea_centered), dim=norm_dim, keepdim=True) # L1 norm across channels # B*1*N / pixels # B*C*1 / pixel and channels * B*1*1
                        elif self.opt.L_norm == 2:
                            fea_norm = torch.norm(fea_centered, dim=norm_dim, keepdim=True) #L2 norm across channels # B*1*N

                        outputs[i]['feature_normalized'] = torch.div(fea_centered, fea_norm)
                        outputs[i]['feature_normalized'] *= self.opt.feat_scale_after_normalize
                # print("fixed:")
                # print(outputs[i]['feature_normalized'])
                # print(losses['final'])
            ###

            if self.opt.zero_edge_region:
                remove_edge(outputs[i]['feature_normalized'])
            ## normalize the max of feature norm of any pixel to 1
            feature = outputs[i]['feature_normalized']
            feature_norm = torch.norm(feature, dim=1, keepdim=True)
            feature_norm = feature_norm / torch.max(feature_norm)
            outputs[i]['feature_norm'] = feature_norm
            
            ## PCA
            if self.opt.pca_in_loss or self.opt.visualize_pca_chnl:
                feature_3, feature_pca = feat_pca(feature)
                outputs[i]['feature_pca'] = feature_pca
                outputs[i]['feature_chnl3'] = feature_3

        return 

    def calc_loss(self, flat_sel, pose1_2, pose1_2_pred=None):
        '''
        Input points are flat
        '''
        ### 1. Calculate gramian of each item
        self.xyz_align_with_pose(flat_sel, pose1_2)

        gramians = {}
        item_to_calc_gram = ['xyz_align', 'feature' ]
        if self.opt.color_in_cost:
            item_to_calc_gram.append('img')

        for item in item_to_calc_gram:
            gramians[item] = self.calc_gramian(flat_sel, item)
        
        ### 2. Calculate inner product
        inner_prods = self.calc_inner_prod(gramians, flat_sel)
        ### 3. Calculate loss
        losses = self.calc_loss_from_inner_prod( inner_prods, flat_sel )

        if self.opt.diff_mode:
            ### 1. Calculate gramian of each item
            gramians_noisy = {}
            gramians_noisy['xyz_noisy'] = self.calc_gramian(flat_sel, 'xyz_noisy')
            gramians_noisy['feature'] = gramians['feature']
            if self.opt.color_in_cost:
                gramians_noisy['img'] = gramians['img']

            ### 2. Calculate inner product
            inner_prods_noisy = self.calc_inner_prod(gramians_noisy, flat_sel)
            ### 3. Calculate loss
            losses_noisy = self.calc_loss_from_inner_prod( inner_prods_noisy, flat_sel )

            return losses, losses_noisy
            
        
        # if self.opt.opt_unet.pose_predict_mode: # obselete
        #     xyz2_trans_pred = torch.matmul(pose1_2_pred, xyz2_homo)[:, 0:3, :] # B*3*N
        #     pcl_diff_exp_pred = kern_mat(xyz1, xyz2_trans_pred, self.dist_coef)
        #     if self.opt.color_in_cost:
        #         inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat * gramian_color )
        #     else:
        #         inner_neg_pred = - torch.mean(pcl_diff_exp_pred * gramian_feat ) 

        #     final_loss = final_loss + inner_neg_pred
        #     loss['CVO_pred'] = inner_neg_pred

        return losses

    def xyz_align_with_pose(self, flat_sel, pose1_2):

        xyz1 = flat_sel[0]['xyz']
        xyz2 = flat_sel[1]['xyz']

        xyz2_homo = torch.cat( ( xyz2, torch.ones((xyz2.shape[0], 1, xyz2.shape[2])).to(self.device) ), dim=1) # B*4*N
        xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
        xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N

        flat_sel[0]['xyz_align'] = flat_sel[0]['xyz']
        flat_sel[1]['xyz_align'] = xyz2_trans

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

            flat_sel[0]['xyz_noisy'] = flat_sel[0]['xyz']
            flat_sel[1]['xyz_noisy'] = xyz2_trans_noisy  
        
        # print('now original')
        # draw3DPts(xyz1, xyz2, flat_sel[0]['img'], flat_sel[1]['img'])
        # print('now matched')
        # draw3DPts(xyz1, xyz2_trans, flat_sel[0]['img'], flat_sel[1]['img'])
        return

    def calc_gramian(self, flat_sel, item):
        gramians = {}
        if self.opt.min_dist_mode:
            list_of_ij = [(0,0), (1,1), (0,1)]
        else:
            list_of_ij =[(0,1)]

        if item == 'xyz_align' or item == 'xyz_noisy':
            item_to_calc_gramian = item
        elif item == 'img':
            self.pre_gramian_img( flat_sel )
            item_to_calc_gramian = 'hsv_graduv' # 'hsv'
        elif item == 'feature':
            item_to_calc_gramian = 'feature_normalized'

        for ij in list_of_ij:
            (i, j) = ij
            if item == 'feature' and (not self.opt.kernalize):
                gramians[ij] = torch.matmul(flat_sel[i][item_to_calc_gramian].transpose(1,2), flat_sel[j][item_to_calc_gramian])
            else:
                gramians[ij] = kern_mat( flat_sel[i][item_to_calc_gramian], flat_sel[j][item_to_calc_gramian], dist_coef=self.dist_coef[item] )
                if torch.isnan(gramians[ij]).any() or torch.isinf(gramians[ij]).any():
                    print("flat_sel i", flat_sel[i][item_to_calc_gramian])
                    print("flat_sel j", flat_sel[j][item_to_calc_gramian])
                    print("feature mean", flat_sel[i]['feature_mean'])
                    print("feature mean", flat_sel[j]['feature_mean'])
                    print("feature norm", flat_sel[i]['feature_norm'])
                    print("feature norm", flat_sel[j]['feature_norm'])
                    print("flat feat i", flat_sel[i]["feature"])
                    print("flat feat j", flat_sel[j]["feature"])
                    print("ij, item:", ij, item, item_to_calc_gramian)
                    print("dist_coef", self.dist_coef[item])
                    print("nan in flat_sel i?", torch.isnan(flat_sel[i][item_to_calc_gramian]).any())
                    print("inf in flat_sel i?", torch.isinf(flat_sel[i][item_to_calc_gramian]).any())
                    print("nan in flat_sel j?", torch.isnan(flat_sel[j][item_to_calc_gramian]).any())
                    print("inf in flat_sel j?", torch.isinf(flat_sel[j][item_to_calc_gramian]).any())
                    
                assert not torch.isnan(gramians[ij]).any()
                assert not torch.isinf(gramians[ij]).any()
                    

        if self.i_batch is not None and self.i_batch % 500 == 0:
            self.print_stat_vec_and_gram(flat_sel, gramians, item_to_calc_gramian)
        return gramians

    def print_stat_vec_and_gram(self, flat_sel, gramians, item):
        for i in range(1):
            print("Vec {} {}, max {}, min {}, mean {}, median {}".format(item, i, flat_sel[i][item].max(), flat_sel[i][item].min(), flat_sel[i][item].mean(), flat_sel[i][item].median() ) )

        if self.opt.min_dist_mode:
            list_of_ij = [(0,0), (0,1)]
        else:
            list_of_ij =[(0,1)]
        for ij in list_of_ij:
            print("Gramian {} {}, max {}, min {}, mean {}, median {}".format(item, ij, gramians[ij].max(), gramians[ij].min(), gramians[ij].mean(), gramians[ij].median() ) )
        print(' ')

    def calc_inner_prod(self, gramians, flat_sel):
        """
        Calculate function distance loss and cosine similarity by multiplying the gramians in all domains together for a specific pair
        """
        inner_prods = {}
        if self.opt.min_dist_mode:
            list_of_ij = [(0,0), (1,1), (0,1)]
        else:
            list_of_ij =[(0,1)]

        for ij in list_of_ij:
            gramian_list = [gramian[ij] for gramian in gramians.values() ]
            n_gram = len(gramian_list)
            
            # gramian_stack = torch.stack(gramian_list, dim=3)
            if self.opt.opt_unet.weight_map_mode:
                (i,j) = ij
                # inner_prods[ij] = torch.sum(torch.prod(gramian_stack, dim=3) * flat_sel[i]['feature_w'].transpose(1, 2) * flat_sel[j]['feature_w'] )
                if n_gram == 1:
                    inner_prods[ij] = torch.sum(gramian_list[0] * flat_sel[i]['feature_w'].transpose(1, 2) * flat_sel[j]['feature_w'] )
                elif n_gram == 2:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] * flat_sel[i]['feature_w'].transpose(1, 2) * flat_sel[j]['feature_w'] )
                elif n_gram == 3:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] * gramian_list[2] * flat_sel[i]['feature_w'].transpose(1, 2) * flat_sel[j]['feature_w'] )
                elif n_gram == 4:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] * gramian_list[2] * gramian_list[3] * flat_sel[i]['feature_w'].transpose(1, 2) * flat_sel[j]['feature_w'] )
                else:
                    raise ValueError("The number of gramian: {} is unexpected".format(n_gram))

            else:
                # inner_prods[ij] = torch.sum(torch.prod(gramian_stack, dim=3) ) # or use mean? 
                if n_gram == 1:
                    inner_prods[ij] = torch.sum(gramian_list[0] )
                elif n_gram == 2:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] )
                elif n_gram == 3:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] * gramian_list[2]  )
                elif n_gram == 4:
                    inner_prods[ij] = torch.sum(gramian_list[0] * gramian_list[1] * gramian_list[2] * gramian_list[3] )
                else:
                    raise ValueError("The number of gramian: {} is unexpected".format(n_gram))

            if self.opt.normalize_inprod_over_pts:
                ni_nj = gramian_list[0].shape[1] * gramian_list[0].shape[2]
                # print('ni_nj', ni_nj)
                inner_prods[ij] = inner_prods[ij] / ni_nj

        return inner_prods

    def calc_loss_from_inner_prod(self, inner_prods, flat_sel):
        losses = {}
        losses['inner_prod'] = inner_prods[(0,1)]
        losses['inner_prod_0_0'] = inner_prods[(0,0)]
        losses['inner_prod_1_1'] = inner_prods[(1,1)]
        
        losses['feat_norm'] = flat_sel[0]['feature_norm_sum'] + flat_sel[1]['feature_norm_sum']
        if self.opt.min_dist_mode:
            losses['func_dist'] = inner_prods[(0,0)] + inner_prods[(1,1)] - 2 * inner_prods[(0,1)]
            losses['cos_sim'] = 1 - inner_prods[(0,1)] / torch.sqrt( inner_prods[(0,0)] * inner_prods[(1,1)] )

        # ############################## penalty on feature norm #####################################
        # # sparsify mode 1 and 6 output norm=0
        # if self.opt.sparsify_mode == 5:
        #     fea_norm_sum_to_loss = fea_norm_sum * 0
        # else:
        #     fea_norm_sum_to_loss = fea_norm_sum * self.norm_scale

        # if not self.opt.opt_unet.weight_map_mode:
        #     final_loss = final_loss + fea_norm_sum_to_loss
        # #############################################################################################
        return losses

    def pre_gramian_img(self, flat_sel):
        for i in range(2):
            # print("rgb_flat, max {}, min {}, mean {}, median {}".format(flat_sel[i]['img'].max(), flat_sel[i]['img'].min(), flat_sel[i]['img'].mean(), flat_sel[i]['img'].median() ) )
            hsv = rgb_to_hsv( flat_sel[i]['img'], flat=True )
            # print("hsv, max {}, min {}, mean {}, median {}".format(hsv.max(), hsv.min(), hsv.mean(), hsv.median() ) )

            # flat_sel[i]['hsv'] = hsv
            flat_sel[i]['hsv_graduv'] = torch.cat((hsv, flat_sel[i]['grad_u'], flat_sel[i]['grad_v']), dim=1)
        return

    def simple_cvo(self, inputs, visualize=False):        
        pose1_2 = inputs['rela_pose']
        kern_size = 5
        sigma = int(kern_size/2)
        smoothing = GaussianSmoothing(1, kern_size, sigma).to(self.device)
        for i in range(2):
            ### mask invalid pixels
            inputs[i]['mask'] = inputs[i]['depth'] > 0
            
            inputs[i]['depth_smooth'] = smoothing(inputs[i]['depth'], padding=True)

            noise_tensor = torch.normal(mean=0, std=1, size=inputs[i]['depth'].shape ).to(self.device)
            noise_tensor = smoothing(noise_tensor, padding=True)
            noise_tensor = smoothing(noise_tensor, padding=True)
            noise_tensor = smoothing(noise_tensor, padding=True)
            inputs[i]['depth_noisy'] = inputs[i]['depth_smooth'] + noise_tensor

        ### flatten
        inout_flat = {}
        inout_flat[0] = {}
        inout_flat[1] = {}
        items_to_be_reshaped = ['depth', 'img', 'mask', 'depth_smooth', 'depth_noisy']
        for i in range(2):
            for item in items_to_be_reshaped:
                if item in inputs[i]:
                    inout_flat[i][item] = inputs[i][item].reshape(inputs[i][item].shape[0], inputs[i][item].shape[1], -1) # B*C*N
                else:
                    print("Inputs: ", list(inputs[i].keys()) )
                    raise ValueError("The item {} to be reshaped is not in either inputs or outputs! ".format(item))

        ### select
        items_to_select = ['depth', 'img', 'depth_smooth', 'depth_noisy']
        batch_size = inputs[i]['img'].shape[0]
        flat_sel = {}
        flat_sel[0] = [None] * batch_size
        flat_sel[1] = [None] * batch_size
        for i in range(2):
            for j in range(batch_size):
                flat_sel[i][j] = {}
                mask = inout_flat[i]['mask'][j].squeeze()
                for item in items_to_select:
                    flat_sel[i][j][item] = inout_flat[i][item][j][:,mask].unsqueeze(0) # B*C*N

                yz1_grid_selected = self.yz1_grid[:,mask]
                xyz_selected = yz1_grid_selected * flat_sel[i][j]['depth'].squeeze(0) # C*N
                flat_sel[i][j]['xyz'] = xyz_selected.unsqueeze(0)                     # B*C*N

                xyz_mean = yz1_grid_selected * flat_sel[i][j]['depth'].mean()
                flat_sel[i][j]['xyz_mean'] = xyz_mean.unsqueeze(0)

                xyz_smooth = yz1_grid_selected * flat_sel[i][j]['depth_smooth'].squeeze(0)
                flat_sel[i][j]['xyz_smooth'] = xyz_smooth.unsqueeze(0)

                xyz_noisy = yz1_grid_selected * flat_sel[i][j]['depth_noisy'].squeeze(0)
                flat_sel[i][j]['xyz_noisy'] = xyz_noisy.unsqueeze(0)

        ### prepare input to cvo
        vectors_to_cvo = {}
        vectors_to_cvo[0] = {}
        vectors_to_cvo[1] = {}
        vectors_to_cvo[0]['xyz'] = flat_sel[0][0]['xyz']
        vectors_to_cvo[0]['rgb'] = flat_sel[0][0]['img']

        weight_mean_grid = np.linspace(0,1,11)
        losses = {}
        losses['func_dist'] = np.zeros(11)
        losses['cos_sim'] = np.zeros(11)
        for k in range(11):
            weight_mean = weight_mean_grid[k]

            vectors_to_cvo[1]['rgb'] = flat_sel[1][0]['img']
            vectors_to_cvo[1]['xyz'] = flat_sel[1][0]['xyz'] * weight_mean + flat_sel[1][0]['xyz_noisy'] * (1-weight_mean) 

            xyz2_homo = torch.cat( ( vectors_to_cvo[1]['xyz'], torch.ones((vectors_to_cvo[1]['xyz'].shape[0], 1, vectors_to_cvo[1]['xyz'].shape[2])).to(self.device) ), dim=1) # B*4*N
            xyz2_trans_homo = torch.matmul(pose1_2, xyz2_homo) # B*4*N
            xyz2_trans = xyz2_trans_homo[:, 0:3, :] # B*3*N
            vectors_to_cvo[1]['xyz'] = xyz2_trans

            ### gramian
            gramians = {}
            items_to_calc_gram = ['xyz', 'rgb']
            list_of_ij = [(0,0), (1,1), (0,1)]
            dist_coef = {}
            dist_coef['xyz'] = 0.1
            dist_coef['rgb'] = 0.1
            for ij in list_of_ij:
                gramians[ij] = {}
                (i,j) = ij
                for item in items_to_calc_gram:
                    gramians[ij][item] = kern_mat( vectors_to_cvo[i][item], vectors_to_cvo[j][item], dist_coef=dist_coef[item] )
            
            ### cvo loss
            inner_prods = {}
            for ij in list_of_ij:
                inner_prods[ij] = torch.sum(gramians[ij]['xyz'] * gramians[ij]['rgb'])

            losses['func_dist'][k] = inner_prods[(0,0)] + inner_prods[(1,1)] - 2 * inner_prods[(0,1)] 
            losses['cos_sim'][k] = 1 - inner_prods[(0,1)] / torch.sqrt( inner_prods[(0,0)] * inner_prods[(1,1)] )

            ### visualize
            if visualize:
                if True: #k == 1 or k == 10:
                    draw3DPts(vectors_to_cvo[0]['xyz'], vectors_to_cvo[1]['xyz'], vectors_to_cvo[0]['rgb'], vectors_to_cvo[1]['rgb'])

        print('loss:', losses['func_dist'])
        plt.figure()
        plt.plot(weight_mean_grid, losses['func_dist'])
        plt.show()