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

def poseMatFromQuatAndT(qw, qx, qy, qz, x, y, z):
    pose_cur = np.matrix(np.identity(4)) 
    pose_cur[0, 3] = x
    pose_cur[1, 3] = y
    pose_cur[2, 3] = z
    pose_cur[0, 0] = 1 - 2*qy*qy - 2*qz*qz
    pose_cur[0, 1] = 2*qx*qy - 2*qz*qw
    pose_cur[0, 2] = 2*qx*qz + 2*qy*qw
    pose_cur[1, 0] = 2*qx*qy + 2*qz*qw
    pose_cur[1, 1] = 1 - 2*qx*qx - 2*qz*qz
    pose_cur[1, 2] = 2*qy*qz - 2*qx*qw
    pose_cur[2, 0] = 2*qx*qz - 2*qy*qw
    pose_cur[2, 1] = 2*qy*qz + 2*qx*qw
    pose_cur[2, 2] = 1 - 2*qx*qx - 2*qy*qy
    return pose_cur

def euler_t_from_pose(pose_mat, transform=None):
    """
    This function generates 4*4 pose matrix in right-handed coordinate.
    Source from CARLA follows a left-handed coordinate(front-right-up), but the definition of rotation is the same as in (front-right-down), 
    which means that a rotation of the same value causes the object to rotate in the same way. It's just that the sign of coordinate
    is defined different. 
    """

    x = pose_mat[0, 3]
    y = pose_mat[1, 3]
    z = pose_mat[2, 3]

    pitch_y_1 = -math.asin(pose_mat[2, 0])
    if pose_mat[2, 0] < 1-1e-5 and pose_mat[2, 0] > -1+1e-5:
        roll_x_1 = math.atan2(pose_mat[2, 1]/math.cos(pitch_y_1), pose_mat[2, 2]/math.cos(pitch_y_1))
        yaw_z_1 = math.atan2(pose_mat[1, 0]/math.cos(pitch_y_1), pose_mat[0, 0]/math.cos(pitch_y_1))
    else:
        yaw_z_1 = 0 # can be anything, gimble lock
        if pose_mat[2, 0] < 0:
            roll_x_1 = yaw_z_1 + math.atan2(pose_mat[0, 1], pose_mat[0, 2])
        else:
            roll_x_1 = -yaw_z_1 + math.atan2(-pose_mat[0, 1], -pose_mat[0, 2])
    
    return np.array([x,y,z,pitch_y_1, roll_x_1, yaw_z_1])

def pose_from_euler_t_Tensor(euler_pose, device, transform=None):
    """
    This function generates 4*4 pose matrix in right-handed coordinate.
    Source from CARLA follows a left-handed coordinate(front-right-up), but the definition of rotation is the same as in (front-right-down), 
    which means that a rotation of the same value causes the object to rotate in the same way. It's just that the sign of coordinate
    is defined different. 
    """
    x = euler_pose[:,0]
    y = euler_pose[:,1]
    z = euler_pose[:,2]
    pitch_y = euler_pose[:,3]
    roll_x = euler_pose[:,4]
    yaw_z = euler_pose[:,5]
    
    if transform == 'Carla':
        z = -z
    batch_num = x.shape[0]
    tensor_list = []
    for i in range(batch_num):
        cy = torch.cos(yaw_z[i] )
        sy = torch.sin(yaw_z[i] )
        cr = torch.cos(roll_x[i] )
        sr = torch.sin(roll_x[i] )
        cp = torch.cos(pitch_y[i] )
        sp = torch.sin(pitch_y[i] )
        # The 4*4 pose matrix is standard (following right-handed coordinate, and all angles are counter-clockwise when positive)
        pose_cur = torch.stack([
            torch.stack([(cp * cy), (cy * sp * sr - sy * cr), (cy * sp * cr + sy * sr), x[i]]),
            torch.stack([(sy * cp), (sy * sp * sr + cy * cr), (-cy * sr + sy * sp * cr), y[i]]), 
            torch.stack([(-sp), (cp * sr), (cp * cr), z[i]]), 
            torch.tensor([0, 0, 0, 1], dtype=torch.float, device=device)
        ]).to(device)
        tensor_list.append(pose_cur)
    return torch.stack(tensor_list, dim=0)

def process_dep_file(dep_file, with_inv=False, source='Carla'):
    assert source == 'Carla' or source == 'TUM', 'error: unrecognized source'
    depth = skimage.io.imread(dep_file)
    if source == 'Carla':
        depth = depth.astype(float) # this step is necessary
        dep_norm = (depth[...,0:1] + depth[...,1:2]*256 + depth[...,2:3]*256*256) /(256*256*256 - 1)
        dep_meter = dep_norm*1000
        dep_sudo_inv = 15/(dep_meter+15)
        # dep_sudo_inv_img = 255/np.amax(dep_sudo_inv) * dep_sudo_inv
    elif source == 'TUM':
        depth = depth.astype(float) # this step is necessary
        depth = depth[..., np.newaxis]
        dep_meter = depth / 5000
        dep_meter_pos = depth / 5000 + 0.001
        # dep_meter[dep_meter==0] = 0.001
        dep_sudo_inv = 1/dep_meter_pos

    if with_inv:
        return dep_meter, dep_sudo_inv
    else:
        return dep_meter

def process_rgb_file(img_file):
    img = skimage.io.imread(img_file)
    img = img.astype(float)
    img_norm = img / 255
    return img_norm

def load_from_carla(folders):
    folders = sorted(folders)

    pair_seq = []
    for folder in folders :
        folder_img = os.path.join(folder, 'CameraRGB')
        files_img = sorted(os.listdir(folder_img) )
        paths_img = [os.path.join(folder_img, f) for f in files_img]

        folder_dep = os.path.join(folder, 'CameraDepth')
        files_dep = sorted(os.listdir(folder_dep) )
        paths_dep = [os.path.join(folder_dep, f) for f in files_dep]

        file_pose = open( os.path.join(folder, 'poses.txt') )
        lines_pose = file_pose.readlines()

        print(len(paths_img))

        assert (len(paths_img) == len(paths_dep) and len(paths_img) <= len(lines_pose) ), "the number of files aren't aligned!"
        
        poses_list = []
        poses_euler_list = []
        for line in lines_pose:
            strs = line.split()
            strs = [float(str_) for str_ in strs]
            x,y,z,pitch,roll,yaw = strs
            pose_mat = pose_from_euler_t(x,y,z,pitch,roll,yaw, transform='Carla')
            poses_list.append(pose_mat)
            poses_euler_list.append( np.array([x,y,z,pitch,roll,yaw]) )

        # # constructing pairs from consequential images
        # for i in range(len(paths_dep)): #range(2): #
        #     frame_num = int( paths_dep[i].split('/')[-1].split('.')[0] )
        #     frame_num_img = int( paths_img[i].split('/')[-1].split('.')[0] )
        #     assert frame_num == frame_num_img, "the names of img and depth are not aligned!"
        #     if i > 0:
        #         pose_relative = np.linalg.inv( poses_list[frame_num_last] ).dot( poses_list[frame_num] )
        #         pair_dict = {'image_path 1': paths_img[i-1], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i-1], 'depth_path 2': paths_dep[i], 'rela_pose': pose_relative}
        #         self.pair_seq.append(pair_dict)
        #     frame_num_last = frame_num

        # constructing pairs from consequential 4 images
        for i in range(len(paths_dep)-1):
            frame_num = int( paths_dep[i].split('/')[-1].split('.')[0] )
            frame_num_img = int( paths_img[i].split('/')[-1].split('.')[0] )
            assert frame_num == frame_num_img, "the names of img and depth are not aligned!"

            for j in range(1, min(5, len(paths_dep)-i) ):
                frame_next = int( paths_dep[i+j].split('/')[-1].split('.')[0] )

                pose_relative = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
                pose_euler_relative = euler_t_from_pose(pose_relative)
                pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                            'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                pair_seq.append(pair_dict)

                pose_relative = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                pose_euler_relative = euler_t_from_pose(pose_relative)
                pair_dict = {'image_path 1': paths_img[i+j], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i+j], 'depth_path 2': paths_dep[i], 
                            'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                pair_seq.append(pair_dict)

        print(folder, 'is loaded')
        # break
    return pair_seq

def load_from_TUM(folders):
    folders = sorted(folders)

    pair_seq = []
    for folder in folders :
        print(folder)

        file_pose = open( os.path.join(folder, 'groundtruth.txt') )
        lines_pose = file_pose.readlines()

        file_match = open(os.path.join(folder, 'match.txt'))
        lines_match = file_match.readlines()

        # read poses
        poses_qt_list = []
        for i, line in enumerate(lines_pose):
            # first three lines are header
            if i < 3:
                continue
            strs = line.split()
            strs = [float(str_) for str_ in strs]
            # t, x, y, z, qx, qy, qz, qw = strs
            poses_qt_list.append(strs)

        print(len(poses_qt_list))

        # read rgb, depth and align it with pose
        paths_img = []
        paths_dep = []
        tstamps_img = []
        poses_list = []
        j_qt = 0
        for line in lines_match:
            strs = line.split()
            tstamp = float(strs[2])
            rgb_file_name = strs[1]
            dep_file_name = strs[3]

            out_of_bound = False
            if poses_qt_list[0][0] >= tstamp:
                # skip until imu time is earlier than current image
                continue

            while poses_qt_list[j_qt][0] < tstamp:
                # print(poses_qt_list[j_qt][0])
                j_qt += 1
                if j_qt == len(poses_qt_list):
                    out_of_bound = True
                    break
            if out_of_bound:
                break

            assert (j_qt != 0), 'error: pose logger starts later than image'

            if poses_qt_list[j_qt][0] == tstamp:
                t, x, y, z, qx, qy, qz, qw = poses_qt_list[j_qt]
            # elif poses_qt_list[j_qt][0] - tstamp > tstamp - poses_qt_list[j_qt-1][0]:
            #     t, x, y, z, qx, qy, qz, qw = poses_qt_list[j_qt-1]
            # elif poses_qt_list[j_qt][0] - tstamp <= tstamp - poses_qt_list[j_qt-1][0]:
            #     t, x, y, z, qx, qy, qz, qw = poses_qt_list[j_qt]
            else:
                # t1, x1, y1, z1, qx1, qy1, qz1, qw1 = poses_qt_list[j_qt]
                # t0, x0, y0, z0, qx0, qy0, qz0, qw0 = poses_qt_list[j_qt-1]
                t0 = poses_qt_list[j_qt-1][0]
                t1 = poses_qt_list[j_qt][0]

                if t1 - t0 > 0.1: 
                    # print('No pose near current time, skipped!')
                    continue

                w0 = (t1 - tstamp)/(t1 - t0)
                w1 = (tstamp - t0)/(t1 - t0)
                
                pose7 = [None]*7
                # ### linear average
                # for j in range(7):
                #     pose7[j] = w0*poses_qt_list[j_qt-1][j+1] + w1*poses_qt_list[j_qt][j+1] 

                ## quaternion average
                # x y z
                for j in range(3):
                    pose7[j] = w0*poses_qt_list[j_qt-1][j+1] + w1*poses_qt_list[j_qt][j+1] 
                
                #qx qy qz qw
                qx0, qy0, qz0, qw0 = poses_qt_list[j_qt-1][4:8]
                qx1, qy1, qz1, qw1 = poses_qt_list[j_qt][4:8]
                wz = math.sqrt( (w0-w1)**2 + 4*w0*w1*(qx0*qx1+qy0*qy1+qz0*qz1+qw0*qw1)**2 )

                p0 = math.sqrt( (w0*(w0-w1+wz))/(wz*(w0+w1+wz)) ) 
                p1 = math.sqrt( (w1*(w1-w0+wz))/(wz*(w0+w1+wz)) )
                if qx0*qx1+qy0*qy1+qz0*qz1+qw0*qw1 > 0:
                    for j in range(3, 7):
                        pose7[j] = p0*poses_qt_list[j_qt-1][j+1] + p1*poses_qt_list[j_qt][j+1]
                else:
                    for j in range(3, 7):
                        pose7[j] = p0*poses_qt_list[j_qt-1][j+1] - p1*poses_qt_list[j_qt][j+1]

                x, y, z, qx, qy, qz, qw = pose7

            pose_mat = poseMatFromQuatAndT(qw, qx, qy, qz, x, y, z)

            if np.linalg.det(pose_mat) > 1.01 or np.linalg.det(pose_mat) < 0.99:
                print('weird!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(qw, qx, qy, qz)
                print(t1, poses_qt_list[j_qt][7], poses_qt_list[j_qt][4], poses_qt_list[j_qt][5], poses_qt_list[j_qt][6] )
                print(t0, poses_qt_list[j_qt-1][7], poses_qt_list[j_qt-1][4], poses_qt_list[j_qt-1][5], poses_qt_list[j_qt-1][6] )
                break
                
                t1 = poses_qt_list[j_qt][0]
                t0 = poses_qt_list[j_qt-1][0]
                pose7 = [None]*7
                for j in range(3):
                    pose7[j] = ( (tstamp - t0)*poses_qt_list[j_qt][j+1] + (t1 - tstamp)*poses_qt_list[j_qt-1][j+1] ) / (t1 - t0)
                for j in range(3, 7):
                    pose7[j] = ( (tstamp - t0)*poses_qt_list[j_qt][j+1] - (t1 - tstamp)*poses_qt_list[j_qt-1][j+1] ) / (t1 - t0)
                x, y, z, qx, qy, qz, qw = pose7
                pose_mat = poseMatFromQuatAndT(qw, qx, qy, qz, x, y, z)

                if np.linalg.det(pose_mat) > 1.01 or np.linalg.det(pose_mat) < 0.99:
                    print('Bad pose skipped!')
                    continue
                
            poses_list.append(pose_mat)

            paths_img.append(os.path.join(folder, rgb_file_name))
            paths_dep.append(os.path.join(folder, dep_file_name))
            tstamps_img.append(tstamp)


        assert (len(paths_img) == len(paths_dep) and len(paths_img) == len(poses_list) ), "the number of rgb and depth files aren't aligned!"

        # constructing pairs from consequential 4 images
        for i in range(len(paths_dep)-1):
            frame_num = i

            for j in range(1, min(10, len(paths_dep)-i), 2 ):
                frame_next = i+j

                pose_relative = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
                pose_euler_relative = euler_t_from_pose(pose_relative)
                pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                            'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                pair_seq.append(pair_dict)

                pose_relative = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                pose_euler_relative = euler_t_from_pose(pose_relative)
                pair_dict = {'image_path 1': paths_img[i+j], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i+j], 'depth_path 2': paths_dep[i], 
                            'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                pair_seq.append(pair_dict)

        print('loaded')
        # break

    return pair_seq

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        image_1, image_2, depth_1, depth_2, idepth_1, idepth_2 = sample['image 1'], sample['image 2'], sample['depth 1'], sample['depth 2'], sample['idepth 1'], sample['idepth 2']
        image_1_raw, image_2_raw = sample['image 1 raw'], sample['image 2 raw']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        depth_1 = depth_1.transpose((2, 0, 1))
        depth_2 = depth_2.transpose((2, 0, 1))
        idepth_1 = idepth_1.transpose((2, 0, 1))
        idepth_2 = idepth_2.transpose((2, 0, 1))

        image_1_raw = image_1_raw.transpose((2, 0, 1))
        image_2_raw = image_2_raw.transpose((2, 0, 1))
        
        if self.device is None:
            return {'image 1': torch.from_numpy(image_1),
                    'image 2': torch.from_numpy(image_2),
                    'image 1 raw': torch.from_numpy(image_1_raw),
                    'image 2 raw': torch.from_numpy(image_2_raw),
                    'depth 1': torch.from_numpy(depth_1),
                    'depth 2': torch.from_numpy(depth_2),
                    'idepth 1': torch.from_numpy(idepth_1),
                    'idepth 2': torch.from_numpy(idepth_2),
                    'rela_pose': torch.from_numpy(sample['rela_pose']),
                    'rela_euler': torch.from_numpy(sample['rela_euler'])}
        else:
            return {'image 1': torch.from_numpy(image_1).to(self.device, dtype=torch.float),
                    'image 2': torch.from_numpy(image_2).to(self.device, dtype=torch.float),
                    'image 1 raw': torch.from_numpy(image_1_raw).to(self.device, dtype=torch.float),
                    'image 2 raw': torch.from_numpy(image_2_raw).to(self.device, dtype=torch.float),
                    'depth 1': torch.from_numpy(depth_1).to(self.device, dtype=torch.float),
                    'depth 2': torch.from_numpy(depth_2).to(self.device, dtype=torch.float),
                    'idepth 1': torch.from_numpy(idepth_1).to(self.device, dtype=torch.float),
                    'idepth 2': torch.from_numpy(idepth_2).to(self.device, dtype=torch.float),
                    'rela_pose': torch.from_numpy(sample['rela_pose']).to(self.device, dtype=torch.float),
                    'rela_euler': torch.from_numpy(sample['rela_euler']).to(self.device, dtype=torch.float)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(72, 96), post_fn=None): # 72, 96
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.post_fn = post_fn

    def __call__(self, sample):
        image_1, image_2, depth_1, depth_2, idepth_1, idepth_2 = sample['image 1'], sample['image 2'], sample['depth 1'], sample['depth 2'], sample['idepth 1'], sample['idepth 2']
        h, w = image_1.shape[:2]

        if isinstance(self.output_size, int):
            # scale the shorter side to output size
            if h > w:
                new_h, new_w = self.output_size / w * h, self.output_size
            else: 
                new_h, new_w = self.output_size, self.output_size / h * w
        else:
            # scale two dimensions respectively
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image_1 = skimage.transform.resize(image_1, (new_h, new_w) )
        image_2 = skimage.transform.resize(image_2, (new_h, new_w) )
        ### different scaling strategy for depth to avoid averaging with zeros to create arrow-like scattered points
        depth_1 = skimage.transform.resize(depth_1, (new_h, new_w), order=0,anti_aliasing=False ) 
        depth_2 = skimage.transform.resize(depth_2, (new_h, new_w), order=0,anti_aliasing=False )
        idepth_1 = skimage.transform.resize(idepth_1, (new_h, new_w) )
        idepth_2 = skimage.transform.resize(idepth_2, (new_h, new_w) )

        if self.post_fn is not None:
            image_1_processed = self.post_fn(image_1)
            image_2_processed = self.post_fn(image_2)

            return {'image 1': image_1_processed,
                    'image 2': image_2_processed,
                    'image 1 raw': image_1,
                    'image 2 raw': image_2,
                    'depth 1': depth_1,
                    'depth 2': depth_2,
                    'idepth 1': idepth_1,
                    'idepth 2': idepth_2,
                    'rela_pose': sample['rela_pose'], 
                    'rela_euler': sample['rela_euler']}
        else:
            return {'image 1': image_1,
                    'image 2': image_2,
                    'image 1 raw': image_1,
                    'image 2 raw': image_2,
                    'depth 1': depth_1,
                    'depth 2': depth_2,
                    'idepth 1': idepth_1,
                    'idepth 2': idepth_2,
                    'rela_pose': sample['rela_pose'], 
                    'rela_euler': sample['rela_euler']}
        



class ImgPoseDataset(Dataset):
    """CARLA images, depth and pose dataset"""

    def __init__(self, root_dir='/mnt/storage/minghanz_data/CARLA(with_pose)/_out', transform=None):
        self.root_dir = root_dir
        if 'CARLA' in root_dir:
            self.folders = glob.glob(root_dir+'/episode*')
            self.pair_seq = load_from_carla(folders=self.folders)
        elif 'TUM' in root_dir:
            self.folders = glob.glob(root_dir+'/rgbd*')
            self.pair_seq = load_from_TUM(folders=self.folders)
        
        self.len_pairs = len(self.pair_seq)

        self.transform = transform

    def __len__(self):
        return self.len_pairs

    def __getitem__(self, idx):
        image_1 = process_rgb_file(self.pair_seq[idx]['image_path 1'])
        image_2 = process_rgb_file(self.pair_seq[idx]['image_path 2'])
        # depth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'])
        # depth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'])
        if 'CARLA' in self.root_dir:
            depth_1, idepth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'], with_inv=True)
            depth_2, idepth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'], with_inv=True)
        elif 'TUM' in self.root_dir:
            depth_1, idepth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'], with_inv=True, source='TUM')
            depth_2, idepth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'], with_inv=True, source='TUM')

        rela_pose = self.pair_seq[idx]['rela_pose']
        rela_euler = self.pair_seq[idx]['rela_euler']

        # sample = {'image 1': image_1, 'image 2': image_2, 'depth 1': depth_1, 'depth 2': depth_2, 'rela_pose': rela_pose}
        sample = {'image 1': image_1, 'image 2': image_2, 'depth 1': depth_1, 'depth 2': depth_2, 'idepth 1': idepth_1, 'idepth 2': idepth_2, 
                    'rela_pose': rela_pose, 'rela_euler': rela_euler}
    
        if self.transform:
            sample = self.transform(sample)

        return sample

def main():
    dataset = ImgPoseDataset(transform=ToTensor(), root_dir = '/mnt/storage/minghanz_data/TUM/RGBD') # root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'
    
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
