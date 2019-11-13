import numpy as np
import os


k_list = [3000]
grid_list = [1]
sample_aug_list = [-1]
output_folder ="/home/minghanz/pytorch-unet/feature_output/with_color_Sat Nov  9 15:00:22 2019"

feat_folder = os.path.join(output_folder, "feature_map")
feat_files = os.listdir(feat_folder)
feat_files = sorted(feat_files)

for k in k_list:
    for g in grid_list:
        for sample_aug in sample_aug_list:
            
            mask_folder = os.path.join(output_folder, 'mask_top_{}/grid_{}/sample_{}'.format(k, g, sample_aug))
            mask_files = os.listdir(mask_folder)
            mask_files = sorted(mask_files)
            for i in range(len(mask_files)):
                mask = np.fromfile(os.path.join(mask_folder, mask_files[i]), dtype=np.bool)
                feat = np.fromfile(os.path.join(feat_folder, feat_files[i]), dtype=np.float32)
                print("feat{}: [{:.3f}, {:.3f}], L1 mean: {}".format(i, 
                    np.min(feat), np.max(feat), np.mean(np.abs(feat)) ) )
                print("feat: ", feat_files[i])