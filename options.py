
class UnetOptions:
    def __init__(self):
        self.in_channels=1
        self.n_classes=2
        self.depth=5
        self.wf=6
        self.padding=False
        self.up_mode='upconv'

        self.batch_norm=False

        self.pose_predict_mode = False
        self.pretrained_mode = False

        self.weight_map_mode = False

        self.continue_train = False

    def setto(self):
        self.in_channels=3
        self.n_classes=16
        self.depth=5
        self.wf=2
        self.padding=True
        self.up_mode='upsample'

class LossOptions:
    def __init__(self, unetoptions):
        self.diff_mode = False
        self.kernalize = True
        
        self.color_in_cost = True

        self.min_dist_mode = True # distance between functions
        self.sparsify_mode = 5 # 1 fix L2 of each pixel, 2 max L2 fix L1 of each channel, 3 min L1, 4 min L2 across channels, 5 fix L1 of each channel, 
                        # 6 no normalization, no norm output

        self.norm_in_loss = False

        self.pca_in_loss = False
        self.subset_in_loss = False

        self.zero_edge_region = True
        
        self.normalize_inprod_over_pts = False

        self.data_aug_rotate180 = False

        self.keep_scale_consistent = True
        self.eval_full_size = False ## This option matters only when keep_scale_consistent is True

        self.trial_mode = False
        self.run_eval = False
        self.top_k_list = [3000]
        self.grid_list = [1]
        self.sample_aug_list = [1, 3, 5, -1]

        self.opt_unet = unetoptions

        if self.run_eval:
            self.folders = ['rgbd_dataset_freiburg1_desk']
        else:
            self.folders = None

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

        self.source='TUM'
        if self.source=='CARLA':
            self.root_dir = '/mnt/storage/minghanz_data/CARLA(with_pose)/_out'
        elif self.source == 'TUM':
            self.root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'

        if self.effective_width > 128:
            self.no_inner_prod = True
        else:
            self.no_inner_prod = False

        
        self.dist_coef = {}
        self.dist_coef['xyz_align'] = 0.1
        self.dist_coef['xyz_noisy'] = 0.1
        self.dist_coef['img'] = 0.5 # originally I used 0.1, but Justin used 0.5 here
        self.dist_coef['feature'] = 0.1 ###?

        self.loss_item = ["cos_sim", "func_dist"] # "cos_sim"
        if self.normalize_inprod_over_pts:
            self.loss_weight = [1, 100]
        else:
            self.loss_weight = [1e6, 1]
        self.lr = 1e-5
        self.lr_decay_iter = 20000
        self.feat_mean_per_chnl = 1e-1 ## the mean abs of a channel across all selected pixels (pre_gramian)
        self.feat_norm_per_pxl = 1 ## the L2 norm of feature vector of a pixel (pre_gramian). 1 means this value has no extra effect

        self.batch_size = 1
        self.effective_batch_size = 4
        self.iter_between_update = int(self.effective_batch_size / self.batch_size)
        self.epochs = 1
        self.total_iters = 20000

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
            
