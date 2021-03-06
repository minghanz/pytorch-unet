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
import copy
import random
import torchsnooper

def skewmat_from_w(w):
    '''
    w is 1-dim torch tensor of length 3
    http://ethaneade.com/lie.pdf
    '''
    zero = torch.tensor(0., dtype=w.dtype, device=w.device)
    skew = torch.stack([
        torch.stack([zero, -w[2], w[1]]), 
        torch.stack([w[2], zero, -w[0]]), 
        torch.stack([-w[1], w[0], zero])
    ])
    return skew

def pose_RtT_from_se3_tensor(w, v, inverse=False):
    b = w.shape[0]
    R = [None]*b
    t = [None]*b
    T = [None]*b
    for i in range(b):
        R[i], t[i], T[i] = pose_RtT_from_se3_tensor_single(w[i], v[i], inverse)

    R = torch.stack(R, dim=0)
    t = torch.stack(t, dim=0)
    T = torch.stack(T, dim=0)
    return R, t, T
    
def pose_RtT_from_se3_tensor_single(w, v, inverse=False):
    '''
    w and v are both 1-dim torch tensor of length 3
    http://ethaneade.com/lie.pdf
    '''
    skew = skewmat_from_w(w)
    theta = torch.norm(w)
    numerical_mode = torch.abs(theta) < 1e-5
    
    I = torch.eye(3, dtype=torch.float, device=w.device)
    
    if numerical_mode:
        R = I
        t = v.unsqueeze(1)
    else:
        A = torch.sin(theta)/theta
        B = ( 1-torch.cos(theta) ) / (theta * theta)
        C = (1 - A) / (theta * theta)

        R = I + A * skew + B * torch.mm(skew, skew)
        V = I + B * skew + C * torch.mm(skew, skew)
        t = torch.mm( V, v.unsqueeze(1) )

    if inverse:
        R = R.transpose(0, 1)
        t = - torch.mm(R, t)

    T = torch.cat([
        torch.cat([R, t], dim=1), 
        torch.tensor([[0, 0, 0, 1]], dtype=torch.float, device=w.device)
    ], dim=0)
    
    return R, t, T

def pose_se3_from_T_tensor(T, pseudo=False):
    ## output B*3, B*3
    b = T.shape[0]
    w = [None]*b
    v = [None]*b
    for i in range(b):
        w[i], v[i] = pose_se3_from_T_tensor_single(T[i], pseudo)

    w = torch.stack(w, dim=0)
    v = torch.stack(v, dim=0)
    return w, v
    
# @torchsnooper.snoop()
def pose_se3_from_T_tensor_single(T, pseudo=False):
    '''
    https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    '''

    R = T[:3, :3]
    t = T[:3, 3]

    cos_theta = (R[0,0] + R[1,1] + R[2,2] - 1) / 2
    theta = torch.acos(cos_theta)
    numerical_mode = torch.isnan(theta).any() or torch.abs(theta) < 1e-5

    if numerical_mode:
        log_R = 1 / 2 * ( R - R.transpose(0, 1) )
    else:
        log_R = theta / ( 2*torch.sin(theta) ) * ( R - R.transpose(0, 1) )

    w = torch.stack( ((log_R[2,1]-log_R[1,2])/2, -(log_R[2,0]-log_R[0,2])/2, (log_R[1,0]-log_R[0,1])/2 ) )

    if not pseudo:
        I = torch.eye(3, dtype=torch.float, device=T.device)
        if numerical_mode:
            # V_inv = I - 0.5 * log_R
            V_inv = I
        else:
            # A1 = ( 1 - theta * torch.cos(theta/2) / ( 2 * torch.sin(theta/2) ) ) / (theta * theta)
            A = 1.0/(theta*theta) - (1+torch.cos(theta))/(2.0*theta*torch.sin(theta))
            # print("A1", A1) # have difference with A1 at 0.000x digit
            # print("A", A)
            V_inv = I - 0.5 * log_R + A * torch.mm(log_R, log_R) 

        v = torch.mm( V_inv, t.unsqueeze(1) )
        v = v.squeeze(1)
    else:
        v = t
    return w, v

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
    gray = skimage.color.rgb2gray(img_norm)
    gray = gray[..., np.newaxis]
    return img_norm, gray

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

        # constructing pairs from consequential 4 images
        for i in range(len(paths_dep)-1):
            frame_num = int( paths_dep[i].split('/')[-1].split('.')[0] )
            frame_num_img = int( paths_img[i].split('/')[-1].split('.')[0] )
            assert frame_num == frame_num_img, "the names of img and depth are not aligned!"

            for j in range(1, min(2, len(paths_dep)-i) ):
                frame_next = int( paths_dep[i+j].split('/')[-1].split('.')[0] )

                pose_relative1_2 = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
                pose_relative2_1 = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )

                pose_euler_relative = euler_t_from_pose(pose_relative1_2)
                pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                            'rela_pose_from_1': pose_relative1_2, 'rela_pose_from_2': pose_relative2_1, 'rela_euler': pose_euler_relative}
                pair_seq.append(pair_dict)

                # pose_relative = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                # pose_euler_relative = euler_t_from_pose(pose_relative)
                # pair_dict = {'image_path 1': paths_img[i+j], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i+j], 'depth_path 2': paths_dep[i], 
                #             'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                # pair_seq.append(pair_dict)

        print(folder, 'is loaded')
        # break
    return pair_seq

def load_from_TUM(folders, simple_mode=False):
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

            if simple_mode:
                for j in range(1,2):
                    frame_next = i+j

                    pose_relative1_2 = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
                    pose_relative2_1 = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                    # pose_relative1_2_res = (pose_relative1_2[:3,:3] + pose_relative1_2.transpose()[:3,:3] )/2
                    # np.fill_diagonal(pose_relative1_2_res, 0)

                    pose_euler_relative = euler_t_from_pose(pose_relative1_2)
                    pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                                'rela_pose_from_1': pose_relative1_2, 'rela_pose_from_2': pose_relative2_1, 'rela_euler': pose_euler_relative}
                    pair_seq.append(pair_dict)

            else:
                for j in range(1, min(2, len(paths_dep)-i) ):
                    frame_next = i+j

                    pose_relative1_2 = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
                    pose_relative2_1 = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                    pose_euler_relative = euler_t_from_pose(pose_relative1_2)
                    pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                                'rela_pose_from_1': pose_relative1_2, 'rela_pose_from_2': pose_relative2_1, 'rela_euler': pose_euler_relative}
                    pair_seq.append(pair_dict)

                    # pose_relative = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
                    # pose_euler_relative = euler_t_from_pose(pose_relative)
                    # pair_dict = {'image_path 1': paths_img[i+j], 'image_path 2': paths_img[i], 'depth_path 1': paths_dep[i+j], 'depth_path 2': paths_dep[i], 
                    #             'rela_pose': pose_relative, 'rela_euler': pose_euler_relative}
                    # pair_seq.append(pair_dict)

        if simple_mode:
            frame_num = len(paths_dep)-1
            i = frame_num
            j = -1
            frame_next = i+j

            pose_relative1_2 = np.linalg.inv( poses_list[frame_num] ).dot( poses_list[frame_next] )
            pose_relative2_1 = np.linalg.inv( poses_list[frame_next] ).dot( poses_list[frame_num] )
            pose_euler_relative = euler_t_from_pose(pose_relative1_2)
            pair_dict = {'image_path 1': paths_img[i], 'image_path 2': paths_img[i+j], 'depth_path 1': paths_dep[i], 'depth_path 2': paths_dep[i+j], 
                        'rela_pose_from_1': pose_relative1_2, 'rela_pose_from_2': pose_relative2_1, 'rela_euler': pose_euler_relative}
            pair_seq.append(pair_dict)

        print('loaded')
        # break

    return pair_seq

class RotateOrNot(object):
    """Rotate 180 degree randomly"""
    def __init__(self, device):
        self.rot180matrix = torch.eye(4).to(device)
        self.rot180matrix[0,0] = -1
        self.rot180matrix[1,1] = -1

    def __call__(self, sample):
        ## https://stackoverflow.com/questions/6824681/get-a-random-boolean-in-python
        rotate_flag = bool( random.getrandbits(1) )

        items_to_rotate = ['img', 'img_raw', 'depth', 'idepth', 'gray']
        if rotate_flag:
            for i in range(2):
                for item in items_to_rotate:
                    sample[i][item] = torch.flip(sample[i][item], dims=[1,2])
            sample['rela_pose_from_1'] = torch.matmul(self.rot180matrix, torch.matmul(sample['rela_pose_from_1'], self.rot180matrix) )
            sample['rela_pose_from_2'] = torch.matmul(self.rot180matrix, torch.matmul(sample['rela_pose_from_2'], self.rot180matrix) )
            #TODO: transform sample['rela_euler'] 
                
        return sample

class SplitBlocks(object):
    """Split the whole image into tiles, each forming an input to the network"""
    def __init__(self, heightwise_num, widthwise_num, effective_h, effective_w):
        self.heightwise_num = heightwise_num
        self.widthwise_num = widthwise_num
        self.effective_h = effective_h
        self.effective_w = effective_w

    def __call__(self, sample):
        samp_split = {}
        samp_split['original'] = copy.deepcopy(sample)
        items_to_split = ['img', 'img_raw', 'depth', 'idepth', 'gray']
        for i in range(self.heightwise_num):
            for j in range(self.widthwise_num):
                samp_split[ (i,j) ] = copy.deepcopy(sample)
                samp_split[ (i,j) ]['ij'] = (i,j)
                h_start = i * self.effective_h
                w_start = j * self.effective_w
                for k in range(2):
                    for item in items_to_split:
                        samp_split[(i,j)][k][item] = sample[k][item][:, h_start:h_start+self.effective_h, w_start:w_start+self.effective_w]

        return samp_split

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # # swap color axis because
        # # numpy image: H x W x C
        # # torch image: C X H X W
        items_to_transpose = ['img', 'img_raw', 'depth', 'idepth', 'gray']
        for i in range(2):
            for item in items_to_transpose:
                trpsed = sample[i][item].transpose( (2, 0, 1) )
                if self.device is None:
                    sample[i][item] = torch.from_numpy(trpsed)
                else:
                    sample[i][item] = torch.from_numpy(trpsed).to(self.device, dtype=torch.float)

        item_to_tensor = ['rela_pose_from_1', 'rela_pose_from_2', 'rela_euler']
        for item in item_to_tensor:
            if self.device is None:
                sample[item] = torch.from_numpy(sample[item])
            else:
                sample[item] = torch.from_numpy(sample[item]).to(self.device, dtype=torch.float)

        return sample

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
        image_1 = sample[0]['img']
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

        items_to_rescale = ['img', 'depth', 'idepth', 'gray']
        items_destination = ['img_raw', 'depth', 'idepth', 'gray']
        for i in range(2):
            for j in range(len(items_to_rescale)):
                mat = sample[i][ items_to_rescale[j] ]
                if items_to_rescale[j] == 'depth':
                    mat = skimage.transform.resize(mat, (new_h, new_w), order=0,anti_aliasing=False )
                else:
                    mat = skimage.transform.resize(mat, (new_h, new_w) )
                sample[i][ items_destination[j] ] = mat
            
            if self.post_fn is not None:
                sample[i]['img'] = self.post_fn( sample[i][ 'img_raw' ] )
            else:
                sample[i]['img'] = sample[i][ 'img_raw' ]

        return sample



class ImgPoseDataset(Dataset):
    """CARLA images, depth and pose dataset"""

    def __init__(self, root_dir='/mnt/storage/minghanz_data/CARLA(with_pose)/_out', transform=None, folders=None):
        self.root_dir = root_dir
        if 'CARLA' in root_dir:
            self.folders = glob.glob(root_dir+'/episode*')
            self.pair_seq = load_from_carla(folders=self.folders)
        elif 'TUM' in root_dir:
            if folders == None:
                self.folders = glob.glob(root_dir+'/rgbd*')
                self.simple = False
            else:
                self.folders = [os.path.join(root_dir, folder) for folder in folders]
                self.simple = True
            self.pair_seq = load_from_TUM(folders=self.folders, simple_mode=self.simple)
        
        self.len_pairs = len(self.pair_seq)

        self.transform = transform

    def __len__(self):
        return self.len_pairs

    def __getitem__(self, idx):
        image_1, gray_1 = process_rgb_file(self.pair_seq[idx]['image_path 1'])
        image_2, gray_2 = process_rgb_file(self.pair_seq[idx]['image_path 2'])
        # depth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'])
        # depth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'])
        if 'CARLA' in self.root_dir:
            depth_1, idepth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'], with_inv=True)
            depth_2, idepth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'], with_inv=True)
        elif 'TUM' in self.root_dir:
            depth_1, idepth_1 = process_dep_file(self.pair_seq[idx]['depth_path 1'], with_inv=True, source='TUM')
            depth_2, idepth_2 = process_dep_file(self.pair_seq[idx]['depth_path 2'], with_inv=True, source='TUM')

        rela_pose_from_1 = self.pair_seq[idx]['rela_pose_from_1']
        rela_pose_from_2 = self.pair_seq[idx]['rela_pose_from_2']
        rela_euler = self.pair_seq[idx]['rela_euler']

        img_name1 = self.pair_seq[idx]['image_path 1'].split('/')[-1][:-4]

        # sample = {'image 1': image_1, 'image 2': image_2, 'depth 1': depth_1, 'depth 2': depth_2, 'rela_pose': rela_pose}
        sample = {}
        sample[0] = {}
        sample[1] = {}
        sample[0].update({'img': image_1, 'gray': gray_1, 'depth': depth_1, 'idepth': idepth_1})
        sample[1].update({'img': image_2, 'gray': gray_2, 'depth': depth_2, 'idepth': idepth_2})

        sample.update({'rela_pose_from_1': rela_pose_from_1, 'rela_pose_from_2': rela_pose_from_2, 'rela_euler': rela_euler, 'imgname 1': img_name1})
    
        if self.transform:
            sample = self.transform(sample)

        return sample

def main():
    dataset = ImgPoseDataset(transform=ToTensor(), root_dir = '/mnt/storage/minghanz_data/TUM/RGBD') # root_dir = '/mnt/storage/minghanz_data/TUM/RGBD'
    
    a = dataset[0]
    depth2 = a[1]['depth'].numpy()
    depth2 = np.squeeze(depth2)
    plt.figure()
    plt.imshow(depth2 )
    image2 = a[1]['img'].numpy()
    image2 = image2.transpose(1,2,0)
    plt.figure()
    plt.imshow(image2)  #.reshape(height, width, channel)
    plt.show()

if __name__ == "__main__":
    main()
