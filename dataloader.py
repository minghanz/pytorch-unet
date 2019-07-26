from __future__ import print_function, division
import os
import torch
# import pandas as pd
import skimage.io, skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import glob
import os
import math

def pose_from_euler_t(x,y,z,pitch_y,roll_x,yaw_z, transform=None):
    """
    This function generates 4*4 pose matrix in right-handed coordinate.
    Source from CARLA follows a left-handed coordinate(front-right-up), but the definition of rotation is the same as in (front-right-down), 
    which means that a rotation of the same value causes the object to rotate in the same way. It's just that the sign of coordinate
    is defined different. 
    """
    if transform == 'Carla':
        z = -z

    cy = math.cos(np.radians(yaw_z ))
    sy = math.sin(np.radians(yaw_z ))
    cr = math.cos(np.radians(roll_x ))
    sr = math.sin(np.radians(roll_x ))
    cp = math.cos(np.radians(pitch_y ))
    sp = math.sin(np.radians(pitch_y ))
    # The 4*4 pose matrix is standard (following right-handed coordinate, and all angles are counter-clockwise when positive)
    pose_cur = np.matrix(np.identity(4)) 
    pose_cur[0, 3] = x
    pose_cur[1, 3] = y
    pose_cur[2, 3] = z
    pose_cur[0, 0] = (cp * cy)
    pose_cur[0, 1] =  (cy * sp * sr - sy * cr)
    pose_cur[0, 2] = (cy * sp * cr + sy * sr)
    pose_cur[1, 0] =  (sy * cp)
    pose_cur[1, 1] = (sy * sp * sr + cy * cr)
    pose_cur[1, 2] = (-cy * sr + sy * sp * cr)
    pose_cur[2, 0] = (-sp)
    pose_cur[2, 1] = (cp * sr)
    pose_cur[2, 2] = (cp * cr)
    return pose_cur

def process_dep_file(dep_file):
    depth = skimage.io.imread(dep_file)
    depth = depth.astype(float) # this step is necessary
    dep_norm = (depth[...,0:1] + depth[...,1:2]*256 + depth[...,2:3]*256*256) /(256*256*256 - 1)
    dep_meter = dep_norm*1000
    dep_sudo_inv = 15/(dep_meter+15)
    # dep_sudo_inv_img = 255/np.amax(dep_sudo_inv) * dep_sudo_inv
    return dep_sudo_inv

def process_rgb_file(img_file):
    img = skimage.io.imread(img_file)
    img = img.astype(float)
    img_norm = img / 255
    return img_norm

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        image_1, image_2, depth_1, depth_2 = sample['image 1'], sample['image 2'], sample['depth 1'], sample['depth 2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        depth_1 = depth_1.transpose((2, 0, 1))
        depth_2 = depth_2.transpose((2, 0, 1))
        
        if self.device is None:
            return {'image 1': torch.from_numpy(image_1),
                    'image 2': torch.from_numpy(image_2),
                    'depth 1': torch.from_numpy(depth_1),
                    'depth 2': torch.from_numpy(depth_2),
                    'rela_pose': torch.from_numpy(sample['rela_pose'])}
        else:
            return {'image 1': torch.from_numpy(image_1).to(self.device, dtype=torch.float),
                    'image 2': torch.from_numpy(image_2).to(self.device, dtype=torch.float),
                    'depth 1': torch.from_numpy(depth_1).to(self.device, dtype=torch.float),
                    'depth 2': torch.from_numpy(depth_2).to(self.device, dtype=torch.float),
                    'rela_pose': torch.from_numpy(sample['rela_pose']).to(self.device, dtype=torch.float)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(84, 112)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_1, image_2, depth_1, depth_2 = sample['image 1'], sample['image 2'], sample['depth 1'], sample['depth 2']
        h, w = image_1.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size / w * h, self.output_size
            else: 
                new_h, new_w = self.output_size, self.output_size / h * w
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_1 = skimage.transform.resize(image_1, (new_h, new_w) )
        image_2 = skimage.transform.resize(image_2, (new_h, new_w) )
        depth_1 = skimage.transform.resize(depth_1, (new_h, new_w) )
        depth_2 = skimage.transform.resize(depth_2, (new_h, new_w) )

        return {'image 1': image_1,
                'image 2': image_2,
                'depth 1': depth_1,
                'depth 2': depth_2,
                'rela_pose': sample['rela_pose']}
        



class ImgPoseDataset(Dataset):
    """CARLA images, depth and pose dataset"""

    def __init__(self, root_dir='/mnt/storage/minghanz_data/CARLA(with_pose)/_out', transform=None):
        self.transform = transform
        self.folders = glob.glob(root_dir+'/episode*')
        self.folders = sorted(self.folders)

        self.pair_seq = []
        for folder in self.folders :
            folder_img = os.path.join(folder, 'CameraRGB')
            files_img = sorted(os.listdir(folder_img) )
            paths_img = [os.path.join(folder_img, f) for f in files_img]

            folder_dep = os.path.join(folder, 'CameraDepth')
            files_dep = sorted(os.listdir(folder_dep) )
            paths_dep = [os.path.join(folder_dep, f) for f in files_dep]

            file_pose = open( os.path.join(folder, 'poses.txt') )
            lines_pose = file_pose.readlines()

            print(len(paths_img))
            print(len(files_dep))
            print(len(lines_pose))

            assert (len(paths_img) == len(paths_dep) and len(paths_img) <= len(lines_pose) ), "the number of files aren't aligned!"
            
            poses_list = []
            for line in lines_pose:
                strs = line.split()
                strs = [float(str_) for str_ in strs]
                x,y,z,pitch,roll,yaw = strs
                pose_mat = pose_from_euler_t(x,y,z,pitch,roll,yaw, transform='Carla')
                poses_list.append(pose_mat)

            for i in range(len(paths_dep)): #range(2): #
                frame_num = int( paths_dep[i].split('/')[-1].split('.')[0] )
                frame_num_img = int( paths_img[i].split('/')[-1].split('.')[0] )
                assert frame_num == frame_num_img, "the names of img and depth are not aligned!"
                if i > 0:
                    pose_relative = np.linalg.inv( poses_list[frame_num_last] ).dot( poses_list[frame_num] )
                    pair_dict = {'image_path 1': paths_img[i-1], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i-1], 'depth_path 2': paths_dep[i], 'rela_pose': pose_relative}
                    self.pair_seq.append(pair_dict)
                frame_num_last = frame_num

            print(folder, 'is loaded')


    def __len__(self):
        return len(self.pair_seq)

    def __getitem__(self, idx):
        image_1 = process_rgb_file(self.pair_seq[idx]['image_path 1'])
        image_2 = process_rgb_file(self.pair_seq[idx]['image_path 2'])
        depth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'])
        depth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'])
        rela_pose = self.pair_seq[idx]['rela_pose']

        sample = {'image 1': image_1, 'image 2': image_2, 'depth 1': depth_1, 'depth 2': depth_2, 'rela_pose': rela_pose}
    
        if self.transform:
            sample = self.transform(sample)

        return sample

def main():
    dataset = ImgPoseDataset(transform=ToTensor()) # 
    
    a = dataset[0]
    depth2 = a['depth 2'].numpy()
    depth2 = np.squeeze(depth2)
    plt.figure()
    plt.imshow(depth2 )
    image2 = a['image 2'].numpy()
    image2 = image2.transpose(1,2,0)
    plt.figure()
    plt.imshow(image2)  #.reshape(height, width, channel)
    plt.show()

if __name__ == "__main__":
    main()
