import open3d as o3d
import numpy as np
import copy

def draw_pcls(source, target=None, uniform_color=False):
    source_temp = copy.deepcopy(source)
    if uniform_color:
        source_temp.paint_uniform_color([1, 0.706, 0])
    if target is not None:
        target_temp = copy.deepcopy(target)
        if uniform_color:
            target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, target_temp])
    else:
        o3d.visualization.draw_geometries([source_temp])
    return

def np_batch_to_o3d_pcd(i, np_batch, color_batch=None):

    pcd = o3d.geometry.PointCloud()

    if color_batch is not None: 
        xyz1_, color_ = thresh_dist_np_batch(i, np_batch, 50, color_batch)
        pcd.points = o3d.utility.Vector3dVector(xyz1_)
        pcd.colors = o3d.utility.Vector3dVector(color_)
    else:
        xyz1_ = thresh_dist_np_batch(i, np_batch, 50)
        pcd.points = o3d.utility.Vector3dVector(xyz1_)

    print('shape of xyz1_:', xyz1_.shape)

    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    return pcd

def thresh_dist_np_batch(i, np_batch, dist_thresh, color_batch=None):
    x1 = np_batch[i, 0, :]
    y1 = np_batch[i, 1, :]
    z1 = np_batch[i, 2, :]

    x1_ = x1[x1<dist_thresh]
    y1_ = y1[x1<dist_thresh]
    z1_ = z1[x1<dist_thresh]

    xyz1_ = np.array([x1_, y1_, z1_]).transpose()

    if color_batch is not None:
        c1 = color_batch[i, 0, :]
        c2 = color_batch[i, 1, :]
        c3 = color_batch[i, 2, :]
        c1_ = c1[x1<dist_thresh]
        c2_ = c2[x1<dist_thresh]
        c3_ = c3[x1<dist_thresh]
        color_ = np.array([c1_, c2_, c3_]).transpose()

        return xyz1_, color_
    else:
        return xyz1_


def draw3DPts(pcl_1, pcl_2=None, color_1=None, color_2=None):
    """
    pcl1 and pcl2 are B*3*N tensors, 3 for x, y, z (front, right, down)
    """
    input_size_1 = list(pcl_1.size() )
    B = input_size_1[0]
    C = input_size_1[1]
    N1 = input_size_1[2]
    if pcl_2 is not None:
        input_size_2 = list(pcl_2.size() )
        N2 = input_size_2[2]

    pcl_1_cpu = pcl_1.cpu().numpy()
    if pcl_2 is not None:
        pcl_2_cpu = pcl_2.cpu().numpy()
    if color_1 is not None:
        color_1_cpu = color_1.cpu().numpy()
    else:
        color_1_cpu = None
    if color_2 is not None:
        color_2_cpu = color_2.cpu().numpy()
    else:
        color_2_cpu = None
    
    
    for i in range(B):
        # fig = plt.figure(i)
        # ax = fig.gca(projection='3d')
        # plt.cla()

        pcd_o3d_1 = np_batch_to_o3d_pcd(i, pcl_1_cpu, color_1_cpu)

        if pcl_2 is not None:
            pcd_o3d_2 = np_batch_to_o3d_pcd(i, pcl_2_cpu, color_2_cpu)
            draw_pcls(pcd_o3d_1, pcd_o3d_2, uniform_color=color_1 is None)
        else:
            draw_pcls(pcd_o3d_1, uniform_color=color_1 is None)

        # plt.axis('equal')
    # plt.show()
    # plt.gca().set_aspect('equal')
    # plt.gca().set_zlim(-10, 10)
    # plt.gca().set_zlim(0, 3.5)