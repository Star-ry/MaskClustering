import os
import open3d as o3d
import numpy as np
from PIL import Image
import cv2
import torch

##### OMNIGIBSON SIMULATION VARIABLES
OMNI_SENSOR_HEIGHT = 1024
OMNI_SENSOR_WIDTH = 1024
OMNI_FOCAL_LENGTH = 17.0
OMNI_HORIZ_APERTURE = 20.954999923706055

DEVICE = 'cuda'

COVERAGE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC = 20
BBOX_EXPAND = 0.1


def get_intrinsic_parameters():
    vert_aperture = OMNI_SENSOR_HEIGHT/OMNI_SENSOR_WIDTH * OMNI_HORIZ_APERTURE
    fx = OMNI_SENSOR_WIDTH * OMNI_FOCAL_LENGTH / OMNI_HORIZ_APERTURE
    fy = OMNI_SENSOR_HEIGHT * OMNI_FOCAL_LENGTH / vert_aperture
    cx = OMNI_SENSOR_HEIGHT * 0.5
    cy = OMNI_SENSOR_WIDTH * 0.5
    return fx, fy, cx, cy


def get_depth(depth_path):
    depth_scale = 1
    depth_map = np.load(depth_path)
    depth = (depth_map * 1000).astype(np.uint16)

    # depth_image = Image.fromarray(depth_scaled)
    # temp_path = 'temp.png'
    # depth_image.save(temp_path)
    depth = depth / depth_scale
    depth = depth.astype(np.float32)
    return depth


def quaternion_rotation_matrix(Q, torch_=False):
    q0, q1, q2, q3 = Q[3], Q[0], Q[1], Q[2]
    
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    if torch_:
        return torch.tensor([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], device=DEVICE)
    else:
        return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=float)


def extrinsic_matrix_torch(c_abs_ori, c_abs_pose):
    rotation = quaternion_rotation_matrix(c_abs_ori, torch_=True)

    x_vector = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).T)
    y_vector = torch.matmul(rotation, torch.tensor([0.0, -1.0, 0.0], device=DEVICE).T)
    z_vector = torch.matmul(rotation, torch.tensor([0.0, 0.0, -1.0], device=DEVICE).T)
    
    rotation_matrix = torch.stack((x_vector, y_vector, z_vector))  # R

    # Translation vector
    translation_vector = -torch.matmul(rotation_matrix, c_abs_pose.view(3, 1))

    # Construct extrinsic matrix RT (4x4)
    RT = torch.eye(4, device=DEVICE)
    RT[:3, :3] = rotation_matrix
    RT[:3, 3] = translation_vector.squeeze()

    # Compute RT inverse (4x4)
    RT_inv = torch.eye(4, device=DEVICE)
    RT_inv[:3, :3] = rotation_matrix.T
    RT_inv[:3, 3] = torch.matmul(rotation_matrix.T, -translation_vector).squeeze()

    return RT.cpu().numpy(), RT_inv.cpu().numpy()


def get_extrinsics(pose_ori_path):
    pose_ori = np.load(pose_ori_path, allow_pickle=True)
    c_abs_pose = torch.tensor(pose_ori[0], dtype=torch.float32, device=DEVICE)
    c_abs_ori = torch.tensor(pose_ori[1], dtype=torch.float32, device=DEVICE)
    RotT, RotT_inv = extrinsic_matrix_torch(c_abs_ori, c_abs_pose)

    return RotT


def get_intrinsics():
    image_size = [1024, 1024]

    intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
    fx, fy, cx, cy = get_intrinsic_parameters()
    intrinisc_cam_parameters.set_intrinsics(image_size[0], image_size[1], fx, fy, cx, cy)
    return intrinisc_cam_parameters


def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC)
    pcld.transform(extrinsics)
    return pcld


def update_vis(vis, points, points_color):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors =  o3d.utility.Vector3dVector(points_color/255)
    vis.add_geometry(points)
    vis.poll_events()
    vis.update_renderer()

if __name__=="__main__":
    extra_info_dir = '/workspace/data/tasmap/capture_test/cook/2a77be2c-c48a-4fd1-b27e-c0153673f884/Kitchen-19107/Kitchen_cook/frames/extra_info'
    intrinisc_cam_parameters = get_intrinsics()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1269,height=778,left=50)
    render_option=vis.get_render_option()
    render_option.point_size=2.0

    for frame_num in os.listdir(extra_info_dir):
        depth_file = os.path.join(extra_info_dir, frame_num, 'depth.npy')
        image_file = os.path.join(extra_info_dir, frame_num, 'original_image.png')
        pose_ori_path = os.path.join(extra_info_dir, frame_num, 'pose_ori.npy')

        depth = get_depth(depth_file)
        extrinsics = get_extrinsics(pose_ori_path)
        pcd = backproject(depth, intrinisc_cam_parameters, extrinsics)

        rgb_image = cv2.imread(image_file)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        points_color = 200
        update_vis(vis, pcd, points_color)