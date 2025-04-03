import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import open3d as o3d
import shutil
from PIL import Image
import imageio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### OMNIGIBSON SIMULATION VARIABLES
OMNI_SENSOR_HEIGHT = 1024
OMNI_SENSOR_WIDTH = 1024
OMNI_FOCAL_LENGTH = 17.0
OMNI_HORIZ_APERTURE = 20.954999923706055
# simulation
TG_PIXEL_X_STRAIGHT = torch.tensor([x for _ in range(OMNI_SENSOR_WIDTH) for x in range(OMNI_SENSOR_HEIGHT)], device=DEVICE)
TG_PIXEL_Y_STRAIGHT = torch.tensor([y for y in range(OMNI_SENSOR_WIDTH) for _ in range(OMNI_SENSOR_HEIGHT)], device=DEVICE)
TG_PIXEL_X = torch.tensor([[x for x in range(OMNI_SENSOR_WIDTH)] for _ in range(OMNI_SENSOR_HEIGHT)], device=DEVICE)
TG_PIXEL_Y = torch.tensor([[y for _ in range(OMNI_SENSOR_WIDTH)] for y in range(OMNI_SENSOR_HEIGHT)], device=DEVICE)

# realsense - real experiment
TG_PIXEL_X_STRAIGHT_REAL = torch.tensor([x for _ in range(848) for x in range(480)], device=DEVICE)
TG_PIXEL_Y_STRAIGHT_REAL = torch.tensor([y for y in range(848) for _ in range(480)], device=DEVICE)
TG_PIXEL_X_REAL = torch.tensor([[x for x in range(848)] for _ in range(480)], device=DEVICE)
TG_PIXEL_Y_REAL = torch.tensor([[y for _ in range(848)] for y in range(480)], device=DEVICE)



def get_intrinsic_parameters(realsense=False):
    if realsense:
    # Realsense d435 카메라 intrinsic matrix
        fx = 605.8658447265625
        fy = 605.128173828125
        cx = 429.753662109375
        cy = 237.18128967285156

    else:
        vert_aperture = OMNI_SENSOR_HEIGHT/OMNI_SENSOR_WIDTH * OMNI_HORIZ_APERTURE

        fx = OMNI_SENSOR_WIDTH * OMNI_FOCAL_LENGTH / OMNI_HORIZ_APERTURE
        fy = OMNI_SENSOR_HEIGHT * OMNI_FOCAL_LENGTH / vert_aperture
        cx = OMNI_SENSOR_HEIGHT * 0.5
        cy = OMNI_SENSOR_WIDTH * 0.5
    return fx, fy, cx, cy




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

    return RT, RT_inv










def copy_file(original_path, output_path):
    # 이미지 복사
    if os.path.exists(original_path):
        shutil.copy(original_path, output_path)
    else:
        print(f"Warning: {original_path} not found.")


def get_depth_image_from_npy(depth_npy_path, depth_image_path):
    depth_map = np.load(depth_npy_path)
    depth_scaled = (depth_map * 1000).astype(np.uint16)

    depth_image = Image.fromarray(depth_scaled)
    depth_image.save(depth_image_path)


def save_tensor_to_txt(tensor, path):
    with open(path, "w") as f:
        for row in tensor:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")





def get_cam_pose(pose_ori_path, output_pose_path):
    pose_ori = np.load(pose_ori_path, allow_pickle=True)
    c_abs_pose = torch.tensor(pose_ori[0], dtype=torch.float32, device=DEVICE)
    c_abs_ori = torch.tensor(pose_ori[1], dtype=torch.float32, device=DEVICE)
    RotT, RotT_inv = extrinsic_matrix_torch(c_abs_ori, c_abs_pose)
    save_tensor_to_txt(RotT_inv, output_pose_path)



def save_mat_to_file(matrix, filename):
    with open(filename, 'w') as f:
        for line in matrix:
            np.savetxt(f, line[np.newaxis], fmt='%f')


def save_intrinsic(mat, intrinsic, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print ('exporting camera intrinsics to', output_path)
    save_mat_to_file(intrinsic, os.path.join(output_path, 'intrinsic_color.txt'))
    save_mat_to_file(mat, os.path.join(output_path, 'extrinsic_color.txt'))
    save_mat_to_file(intrinsic, os.path.join(output_path, 'intrinsic_depth.txt'))
    save_mat_to_file(mat, os.path.join(output_path, 'extrinsic_depth.txt'))


def save_2D(extra_info_path, output_2D_dir):

    os.makedirs(os.path.join(output_2D_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_2D_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_2D_dir, 'depth_npy'), exist_ok=True)
    os.makedirs(os.path.join(output_2D_dir, 'pose'), exist_ok=True)
    os.makedirs(os.path.join(output_2D_dir, 'intrinsic'), exist_ok=True)

    for frame in tqdm(sorted(os.listdir(extra_info_path)), desc="TASMap 2D", leave=False):

        image_path = os.path.join(extra_info_path, frame, 'original_image.png')
        output_image_path = os.path.join(output_2D_dir, 'color', f'{frame}.jpg')
        copy_file(image_path, output_image_path)


        depth_npy_path = os.path.join(extra_info_path, frame, 'depth.npy')
        output_depth_npy_path = os.path.join(output_2D_dir, 'depth_npy', f'{frame}.npy')
        copy_file(depth_npy_path, output_depth_npy_path)

        depth_path = os.path.join(extra_info_path, frame, 'depth_image.png')
        output_depth_image_path = os.path.join(output_2D_dir, 'depth', f'{frame}.png')
        get_depth_image_from_npy(depth_npy_path, output_depth_image_path)

        pose_ori_path = os.path.join(extra_info_path, frame, 'pose_ori.npy')
        output_pose_path = os.path.join(output_2D_dir, 'pose', f'{frame}.txt')
        get_cam_pose(pose_ori_path, output_pose_path)

    output_intrinsic_path = os.path.join(output_2D_dir, 'intrinsic')
    ext_mat = np.eye(4)
    fx, fy, cx, cy = get_intrinsic_parameters(realsense=False)
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    save_intrinsic(ext_mat, K, output_intrinsic_path)

def get_depth(depth_path):
    depth = cv2.imread(depth_path, -1)
    depth = depth / depth_scale
    depth = depth.astype(np.float32)
    return depth


def get_intrinsics(image_size, intrinsic_path):
    intrinsics = np.loadtxt(intrinsic_path)

    intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
    intrinisc_cam_parameters.set_intrinsics(image_size[0], image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    return intrinisc_cam_parameters


def read_pose(path):
    return np.loadtxt(path)


def backproject(color, depth, intrinisc_cam_parameters, extrinsics):
    """
    Convert RGB and depth images to a colored point cloud.
    """
    DEPTH_LIMIT = 20
    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=DEPTH_LIMIT,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinisc_cam_parameters
    )
    pcd.transform(extrinsics)
    return pcd


def create_downsampled_point_cloud(base_dir, image_size=(1024, 1024), stride=1, depth_trunc=0.005, buffer_size=10, depth_scale=1000):
    depth_dir = os.path.join(base_dir, 'depth')
    color_dir = os.path.join(base_dir, 'color')
    pose_dir = os.path.join(base_dir, 'pose')
    intr_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')

    image_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('.')[0]))

    frame_ids = [x.split('.')[0] for x in image_list][::stride]
    intrinisc_cam_parameters = get_intrinsics(image_size, intr_path)

    full_pcd = o3d.geometry.PointCloud()
    buffer_pcd = o3d.geometry.PointCloud()

    with tqdm(total=len(frame_ids)) as pbar:
        for i, fid in enumerate(frame_ids):
            pbar.set_description(f"Pointcloud (Frame {fid})")
            pbar.update(1)

            depth_path = os.path.join(depth_dir, f"{fid}.png")
            depth = get_depth(depth_path)
            color = imageio.v2.imread(os.path.join(color_dir, f"{fid}.jpg"))

            # Resize color image to match depth image shape
            if color.shape[:2] != depth.shape:
                color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            extrinsics = read_pose(os.path.join(pose_dir, f"{fid}.txt"))

            pcd = backproject(color, depth, intrinisc_cam_parameters, extrinsics)
            buffer_pcd += pcd

            if (i + 1) % buffer_size == 0:
                buffer_pcd = buffer_pcd.voxel_down_sample(voxel_size=depth_trunc)
                full_pcd += buffer_pcd
                buffer_pcd.clear()

        # Merge any remaining points in buffer
        if len(buffer_pcd.points) > 0:
            buffer_pcd = buffer_pcd.voxel_down_sample(voxel_size=depth_trunc)
            full_pcd += buffer_pcd
            buffer_pcd.clear()

    full_pcd = full_pcd.voxel_down_sample(voxel_size=depth_trunc)

    return full_pcd



if __name__=="__main__":

    # scene_path = '/workspace/data/tasmap/capture_test/cook/2a77be2c-c48a-4fd1-b27e-c0153673f884/Kitchen-19107/Kitchen_cook/frames/extra_info'
    # scene_name = 'scene0000_00'

    # scene_path = '/workspace/data/tasmap/capture_test/cook/2a77be2c-c48a-4fd1-b27e-c0153673f884/LivingDiningRoom-19050/LivingDiningRoom_cook/frames/extra_info'
    # scene_name = 'scene0001_00'

    scene_path = '/workspace/data/tasmap/capture_test/cook/2a77be2c-c48a-4fd1-b27e-c0153673f884/MasterBedroom-19001/MasterBedroom_cook/frames/extra_info'
    scene_name = 'scene0002_00'

    data_root_dir = os.path.join('/workspace/MaskClustering/data/tasmap', 'processed')
    data_dir = os.path.join(data_root_dir, scene_name)
    os.makedirs(data_dir, exist_ok=True)

    # save 2D images
    save_2D(scene_path, data_dir)

    # pointcloud settings
    output_ply_path = os.path.join(data_dir, f'{scene_name}_vh_clean_2.ply')
    image_size = (OMNI_SENSOR_WIDTH, OMNI_SENSOR_HEIGHT)
    # image_size = (640, 480)
    depth_scale = 1000
    stride = 1
    depth_trunc=0.005
    buffer_size = 30

    # save downsampled ply
    pcd = create_downsampled_point_cloud(data_dir, image_size=image_size, stride=stride, depth_trunc=depth_trunc, buffer_size=buffer_size, depth_scale=depth_scale)
    o3d.visualization.draw_geometries([pcd], window_name="Downsampled Point Cloud")
    o3d.io.write_point_cloud(output_ply_path, pcd)