import open3d as o3d
import numpy as np
import copy

def draw_pcls(source, target=None):
    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    if target is not None:
        target_temp = copy.deepcopy(target)
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, target_temp])
    else:
        o3d.visualization.draw_geometries([source_temp])
    return

def np_batch_to_o3d_pcd(np_batch, i):

    x1 = np_batch[i, 0, :]
    y1 = np_batch[i, 1, :]
    z1 = np_batch[i, 2, :]

    x1_ = x1[x1<200]
    y1_ = y1[x1<200]
    z1_ = z1[x1<200]

    xyz1_ = np.array([x1_, y1_, z1_]).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz1_)
    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    return pcd