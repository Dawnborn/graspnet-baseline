""" Modules for GraspNet baseline model.
    Author: chenxi-wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pointnet2.pytorch_utils as pt_utils
from pointnet2.pointnet2_utils import CylinderQueryAndGroup
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2+self.num_view)

    def forward(self, seed_xyz, seed_features, end_points):
        """ Forward pass.
                将接近向量的预测转化为一个分类问题，每个seed点按照预定义的方式生成300个view（每个viewpoint对应一个接近方向），每个view预测一个confidence
                同时每个seed点计算一个2dim的objectness score，表示这个位置可不可抓的二分类confidence
                并返回每个seed_point最高得分view的位姿和id
            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
                objectness_score: (B, 2, num_seed) 
                view_score: (B, num_seed, num_view)
                grasp_top_view_inds: (B, num_seed)
                grasp_top_view_score: (B, num_seed)
                grasp_top_view_xyz: (B, num_seed, 3)
                grasp_top_view_rot: (B, num_seed, 3, 3)
        """
        # 将 seed_features 转化为2位的objectness_score和num_view维度的viewscore
        B, num_seed, _ = seed_xyz.size()
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        objectness_score = features[:, :2, :] # (B, 2, num_seed) <- 1 302 1024 两个维度表示没有或有物体
        view_score = features[:, 2:2+self.num_view, :].transpose(1,2).contiguous() # (B, num_seed, num_view) 1 1024 300
        end_points['objectness_score'] = objectness_score
        end_points['view_score'] = view_score

        # print(view_score.min(), view_score.max(), view_score.mean())
        top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed) # 
        top_view_inds_= top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        template_views = generate_grasp_views(self.num_view).to(features.device) # (num_view, 3) 按照斐波那契格子生成viewpoint在单位圆上
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3) # 选择最高分对应的iewpoint
        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device) # num_seed
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot

        return end_points


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]
        
        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot):
        """ Forward pass.
                对于每一个seedpoint将点云转换到该seedpoint为原点，接近向量为x轴的坐标系下
                按照不同的抓取深度定义num_depths个以该seedpoint为原点，轴向为接近向量的圆柱，
                对于每一个圆柱，从点云中采样n_sample个在圆柱里的点(batch_size, 3, num_seed, num_depth, nsample)
                然后经过mlp提取特征，结果为每个seedpoint提供num_depth个local特征 (B, -1, num_seed, num_depth)
            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,dim_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers) # 4
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            )) # (batch_size, feature_dim, num_seed, nsample) # 1 3 1024 64 feature为旋转后的点坐标
        grouped_features = torch.stack(grouped_features, dim=3) # -> (batch_size, feature_dim, num_seed, num_depth, nsample) 1 3 1024 4 64
        grouped_features = grouped_features.view(B, -1, num_seed*num_depth, self.nsample) # (batch_size, feature_dim, num_seed*num_depth, nsample) torch.Size([1, 3, 4096, 64])

        vp_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth) # 1 256 1024 4
        return vp_features

        
class OperationNet(nn.Module):
    """ Grasp configure estimation.
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3*num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.
                将平面内的抓取角度转化为12个预定义角度下的分类问题，输入每个seed每个抓取深度的feature，通过1d卷积转换成12个抓取角度angle下的confidence，各个抓取角度下的grasp score,和各个抓取角度下的width
                    抓取得分预测（grasp_score_pred）：grasp score定义方式为力闭合
                    抓取角度类别预测（grasp_angle_cls_pred）：每个角度的confidence
                    抓取宽度预测（grasp_width_pred）：这部分输出预测了在每个角度类别下，夹爪应该开启的宽度。
            Input:
                vp_features: [torch.FloatTensor, torch.Size([B, dim_feature, num_seed, num_depth])]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points['grasp_score_pred']  (1 self.num_angle num_seed num_depth)
                end_points['grasp_angle_cls_pred'] (1 self.num_angle num_seed num_depth)
                end_points['grasp_width_pred'] (1 self.num_angle num_seed num_depth)
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth) # 1 3*self.num_angle 1024 4

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle] # torch.Size([1, 12, 1024, 4])
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2*self.num_angle:3*self.num_angle]
        return end_points

    
class ToleranceNet(nn.Module):
    """ Grasp tolerance prediction.
    
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_deph = num_depth
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.
                预测每个抓取点和在每个抓取深度下的tolerance，物理含义为该grasp在多大的pose误差下依然能保证一定抓取效果
            Input:
                vp_features: [torch.FloatTensor, (batch_size, dim, num_seed, num_depth)] torch.Size([1, 256, 1024, 4])
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
                grasp_tolerance_pred: (B, -1, num_seed, num_depth) -> torch.Size([1, 12, 1024, 4]) 约5e-3
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points