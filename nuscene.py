from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from collections import OrderedDict
import numpy as np
import yaml
import os

# 初始化 nuScenes mini 数据集
#TODO: 这里是mini数据集，后面换成v1
data_root = r"C:\Users\Guoji\Desktop\Files\data\mini"
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

# 相机通道名
camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# 创建输出文件夹
output_dir = 'camera_configs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化每个相机的结构
camera_data_dict = {cam: {
    'H':None,
    'W':None,
    'intrinsic': None,
    'extrinsic': None,
    'frames': {}  # key: sample_token, value: dict with ego_pose and camera_pose
} for cam in camera_channels}

# 遍历所有 sample
for sample in nusc.sample:
    sample_token = sample['token']

    for cam in camera_channels:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)

        # 1. 获取 calibration（只保存一次）
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        if camera_data_dict[cam]['intrinsic'] is None:
            camera_data_dict[cam]['H'] = cam_data['height']
            camera_data_dict[cam]['W'] = cam_data['width']
            camera_data_dict[cam]['intrinsic'] = calib['camera_intrinsic']
            extrinsic = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=False)
            camera_data_dict[cam]['extrinsic'] = extrinsic.tolist()
            cam_data = nusc.get('sample_data', cam_token)

        # 2. 获取 ego pose
        ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_matrix = transform_matrix(ego['translation'], Quaternion(ego['rotation']), inverse=False)

        # 3. 相机 pose（global 中） = ego_pose @ camera_to_ego
        extrinsic = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=False)
        camera_pose = ego_matrix @ extrinsic

        # 4. 保存该帧的 pose 数据
        camera_data_dict[cam]['frames'][sample_token] = {
            'ego_pose': ego_matrix.tolist(),
            'camera_pose': camera_pose.tolist()
        }

# 写入每个相机的 YAML 文件
for cam in camera_channels:
    output_file = os.path.join(output_dir, f'{cam}.yaml')
    ordered_data = OrderedDict()
    ordered_data['H'] = camera_data_dict[cam]['H']
    ordered_data['W'] = camera_data_dict[cam]['W']
    ordered_data['intrinsic'] = camera_data_dict[cam]['intrinsic']
    ordered_data['extrinsic'] = camera_data_dict[cam]['extrinsic']
    ordered_data['frames'] = camera_data_dict[cam]['frames']

    with open(output_file, 'w') as f:
        yaml.dump(ordered_data, f)


