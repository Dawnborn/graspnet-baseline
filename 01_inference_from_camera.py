""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.graspnet import GraspNet, pred_decode      # 网络
from dataset.graspnet_dataset import GraspNetDataset        # 数据集
from utils.collision_detector import ModelFreeCollisionDetector       # 碰撞检查
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image      # 深度图转点云

import pyrealsense2 as rs
import matplotlib.pyplot as plt
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default="/data/hdd1/storage/junpeng/ws_anygrasp/graspnet-baseline/ckpt/checkpoint-rs.tar", help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir, color, depth):
    # load data
    # color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    color = color / 255.0
    # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    # load data from camera
    # intrinsic = meta['intrinsic_matrix']
    # print(intrinsic)
    # factor_depth = meta['factor_depth']
    # print(factor_depth)
    intrinsic = np.array([894.702 , 0.,         632.479,
                          0. ,        894.702, 359.741,
                          0.,           0.,           1.        ]).reshape((3,3))
    factor_depth = 1000.0



    # generate cloud
    # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) # 720，1280，3

    # get valid points
    mask = (workspace_mask & (depth > 0))
    print(np.shape((mask)))
    print(np.shape(color))
    print(np.shape(cloud))

    cloud_masked = cloud[mask]
    color_masked = color[mask]
    print(np.shape(color_masked))
    print(np.shape(cloud_masked))

    # plt.imshow(mask)
    # plt.show()
    # sample points
    if len(cloud_masked) >= cfgs.num_point: # mask超过cfg所需点数则无放回选择
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else: # 若mask后点数少于所需点数则先全选，然后重复choice
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    cfg = pipeline.start(config)
    for i in range(10):
    # while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        # depth = frames.get_depth_frame()
        # color = frames.get_color_frame()
        depth = aligned_frames.get_depth_frame() # 720 1280 0-21401
        color = aligned_frames.get_color_frame() # 720 1280 0-256
        depth = np.asanyarray(depth.get_data())
        color = np.asanyarray(color.get_data())

        from datetime import datetime
        # 获取当前时间
        now = datetime.now()
        # 格式化时间
        formatted_time = now.strftime("%Y%m%d%H%M")
        depth_path = os.path.join(data_dir,"depth_{}.png".format(i))
        color_path = os.path.join(data_dir,"color_{}.png".format(i))
        cv2.imwrite(depth_path, depth)
        plt.imsave(color_path, color)

        end_points, cloud = get_and_process_data(data_dir, color, depth) # endpoint 包含pointcloud map和颜色
        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    # data_dir = "/data/hdd1/storage/junpeng/ws_anygrasp/graspnet-baseline/doc/example_data"
    # data_dir = '/data/hdd1/storage/junpeng/ws_anygrasp/graspnet-baseline/test_data'
    demo(data_dir)
