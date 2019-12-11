
class UnetOptions:
    def __init__(self):
        self.in_channels=1
        self.n_classes=2
        self.depth=5
        self.wf=6
        self.padding=False
        self.up_mode='upconv'

        self.batch_norm=True
        self.non_neg = True

        self.pose_predict_mode = False
        self.pretrained_mode = False

        self.weight_map_mode = False

    def setto(self):
        self.in_channels=3
        self.n_classes=6
        self.depth=5
        self.wf=2
        self.padding=True
        self.up_mode='upsample'


## TODO: normalizing using std
## TODO: changing length scale
## TODO: batch norm

class LossOptions:
    def __init__(self, unetoptions):
        
        self.color_in_cost = False

        # self.diff_mode = False
        # self.min_grad_mode = True
        # self.min_dist_mode = True # distance between functions

        # # self.kernalize = True
        # self.sparsify_mode = 2 # 1 fix L2 of each pixel, 2 max L2 fix L1 of each channel, 3 min L1, 4 min L2 across channels, 5 fix L1 of each channel, 
        #                 # 6 no normalization, no norm output
        # self.L_norm = 2 # 1, 2, (1,2) #when using (1,2), sparsify mode 2 or 5 are the same
        # ## before 10/23/2019, use L_norm=1, sparsify_mode=5, kernalize=True, batch_norm=False, dist_coef = 0.1
        # ## L_norm=1, sparsify_mode=2, kernalize=True, batch_norm=False, dist_coef = 0.1, feat_scale_after_normalize = 1e-1
        # ## L_norm=2, sparsify_mode=2, kernalize=True, batch_norm=False, dist_coef = 0.1, feat_scale_after_normalize = 1e1

        self.norm_in_loss = False
        self.pca_in_loss = False
        self.subset_in_loss = False

        self.zero_edge_region = True
        self.normalize_inprod_over_pts = False

        self.data_aug_rotate180 = True
        self.keep_scale_consistent = False
        self.eval_full_size = False ## This option matters only when keep_scale_consistent is True

        self.test_no_log = False
        self.trial_mode = False
        self.run_eval = False
        self.continue_train = False
        self.visualize_pcd = False
        self.top_k_list = [3000]
        self.grid_list = [1]
        self.sample_aug_list = [-1]

        self.opt_unet = unetoptions

        self.dist_coef = {}
        self.dist_coef['xyz_align'] = 0.2 # 0.1
        self.dist_coef['xyz_noisy'] = 0.2 # 0.1
        self.dist_coef['xyz'] = 0.2 # 0.1
        self.dist_coef['img'] = 0.5 # originally I used 0.1, but Justin used 0.5 here
        # self.dist_coef['feature'] = 0.1 ###?
        self.dist_coef_feature = 0.1

        self.lr = 1e-5
        self.lr_decay_iter = 20000

        self.batch_size = 1
        self.effective_batch_size = 2
        self.iter_between_update = int(self.effective_batch_size / self.batch_size)
        self.epochs = 1
        self.total_iters = 20000


    def set_loss_from_options(self):

        self.kernalize = not self.opt_unet.non_neg

        if self.keep_scale_consistent:
            self.width = 640 
            self.height = 480
            if self.run_eval:
                self.height_split = 1
                self.width_split = 1
            else:
                self.height_split = 5
                self.width_split = 5
            self.effective_width = int(self.width / self.width_split)
            self.effective_height = int(self.height / self.height_split)
        else:
            if self.run_eval:
                self.width = 640 
                self.height = 480
            else:
                self.width = 128 # (72*96) [[96, 128]] (240, 320)
                self.height = 96
            self.effective_width = self.width 
            self.effective_height = self.height 

        if self.effective_width > 128:
            self.no_inner_prod = True
        else:
            self.no_inner_prod = False

        self.source='TUM'
        if self.source=='CARLA':
            self.root_dir = '/mnt/storage/minghanz_data/CARLA(with_pose)/_out'
        elif self.source == 'TUM':
            # self.root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'
            # self.root_dir = '/media/minghanz/Seagate_Backup_Plus_Drive/TUM/rgbd_dataset/untarred'
            self.root_dir = '/home/minghanz/Datasets/TUM/rgbd/untarred'

        # if self.run_eval or self.continue_train:
        #     self.pretrained_weight_path = 'saved_models/with_color_Fri Nov  8 22:21:30 2019/epoch00_20000.pth'

        if self.run_eval:
            self.folders = ['rgbd_dataset_freiburg1_desk']
        else:
            self.folders = None

        # if self.L_norm == 1:
        #     self.feat_scale_after_normalize = 1e-2 ## the mean abs of a channel across all selected pixels (pre_gramian)
        # elif self.L_norm == 2:
        #     self.feat_scale_after_normalize = 1e1
        # elif self.L_norm == (1,2):
        #     self.feat_scale_after_normalize = 1e-1
        if self.kernalize: # rbf
            self.sparsify_mode = 2 # norm_dim=2, centralize, normalize
            self.L_norm = 2 # make variance 1 of all channels
            self.feat_scale_after_normalize = 1e-1 # sdv 1e-1
            self.reg_norm_mode = False
        else: # dot product
            self.sparsify_mode = 2 # 3, norm_dim=2, centralize, doesn't normalize (use the norm as loss)
            self.L_norm = (1,2) # (1,2) # mean L2 norm of all pixel features
            if self.self_sparse_mode:
                self.feat_scale_after_normalize = 1e-1
            else:
                self.feat_scale_after_normalize = 1e0
            self.reg_norm_mode = False

        self.feat_norm_per_pxl = 1 ## the L2 norm of feature vector of a pixel (pre_gramian). 1 means this value has no extra effect

        self.set_loss_from_mode()

        return
    def set_loss_from_mode(self):
        ## loss function setting
        self.loss_item = []
        self.loss_weight = []
        if self.min_dist_mode:
            self.loss_item.extend(["cos_sim", "func_dist"])
            if self.normalize_inprod_over_pts:
                self.loss_weight.extend([1, 100])
            else:
                if self.kernalize:
                    self.loss_weight.extend([1, 1e-5]) # 1e6, 1 # 1e-4: dots, 1e-6: blocks
                elif self.self_trans:
                    self.loss_weight.extend([0,-1e-6])
                else:
                    self.loss_weight.extend([1, 1e-6]) # 1e6, 1
        if self.diff_mode:
            self.loss_item.extend(["cos_diff", "dist_diff"])
            if self.normalize_inprod_over_pts:
                self.loss_weight.extend([1, 100])
            else:
                if self.kernalize:
                    self.loss_weight.extend([1, 1e-6]) # 1e6, 1
                else:
                    self.loss_weight.extend([1, 1e-4]) # 1e6, 1
        if self.min_grad_mode:
            self.loss_item.extend(["w_angle", "v_angle"])
            self.loss_weight.extend([1, 1])
        if self.reg_norm_mode:
            self.loss_item.append("feat_norm")
            self.loss_weight.append(1e-1) # self.reg_norm_weight

        if self.self_sparse_mode:
            self.loss_item.extend(["cos_sim", "func_dist"])
            self.loss_weight.extend([0, 1]) # 1e6, 1

        self.samp_pt = self.min_grad_mode
        return

    def set_from_manual(self, overwrite_opt):
        manual_keys = overwrite_opt.__dict__.keys() # http://www.blog.pythonlibrary.org/2013/01/11/how-to-get-a-list-of-class-attributes/
        for key in manual_keys:
            vars(self)[key] = vars(overwrite_opt)[key] # https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
        self.dist_coef['feature'] = self.dist_coef_feature
        return

    def set_eval_full_size(self):
        if self.eval_full_size == False:
            self.eval_full_size = True
            self.no_inner_prod_ori = self.no_inner_prod
            self.visualize_pca_chnl_ori = self.visualize_pca_chnl
            self.no_inner_prod = True
            self.visualize_pca_chnl = False

    def unset_eval_full_size(self):
        if self.eval_full_size == True:
            self.eval_full_size = False
            self.no_inner_prod = self.no_inner_prod_ori
            self.visualize_pca_chnl = self.visualize_pca_chnl_ori
            
