
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
        
        self.width = 128 # (72*96) (96, 128) (240, 320)
        self.height = 96
        self.source='TUM'
        if self.source=='CARLA':
            self.root_dir = '/mnt/storage/minghanz_data/CARLA(with_pose)/_out'
        elif self.source == 'TUM':
            self.root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'

        self.min_dist_mode = True # distance between functions
        self.sparsify_mode = 5 # 1 fix L2 of each pixel, 2 max L2 fix L1 of each channel, 3 min L1, 4 min L2 across channels, 5 fix L1 of each channel, 
                        # 6 no normalization, no norm output

        self.pca_in_loss = False
        self.subset_in_loss = True
        
        self.opt_unet = unetoptions
