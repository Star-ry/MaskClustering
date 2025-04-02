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


def intrinsic_matrix_torch(realsense=False):
    fx, fy, cx, cy = get_intrinsic_parameters(realsense)

    K = torch.tensor([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], device=DEVICE)

    K_inv = torch.inverse(K)

    return K, K_inv


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


# def extrinsic_matrix_torch(c_abs_ori, c_abs_pose):
#     rotation = quaternion_rotation_matrix(c_abs_ori, torch_=True)

#     x_vector = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).T)
#     y_vector = torch.matmul(rotation, torch.tensor([0.0, -1.0, 0.0], device=DEVICE).T)
#     z_vector = torch.matmul(rotation, torch.tensor([0.0, 0.0, -1.0], device=DEVICE).T)
    
#     rotation_matrix = torch.stack((x_vector, y_vector, z_vector))

#     rotation_matrix_inv = torch.inverse(rotation_matrix)

#     transition_vector = -1 * torch.matmul(rotation_matrix, c_abs_pose.float().T).T
    
#     RT = torch.cat((rotation_matrix, torch.tensor([[transition_vector[0]],[transition_vector[1]],[transition_vector[2]]], device=DEVICE)), dim=1)
#     RT = torch.cat((RT, torch.tensor([[0, 0, 0, 1]], device=DEVICE)), dim=0)

#     RT_inv = torch.cat((rotation_matrix_inv, torch.tensor([[c_abs_pose[0]], [c_abs_pose[1]], [c_abs_pose[2]]], device=DEVICE)), dim=1)
#     RT_inv = torch.cat((RT_inv, torch.tensor([[0, 0, 0, 1]], device=DEVICE)), dim=0)

#     return RT, RT_inv

def get_extrinsic(ori, pos):
    # ori: [qx, qy, qz, qw] (쿼터니언), pos: [tx, ty, tz]
    qx, qy, qz, qw = ori
    # qw, qx, qy, qz = ori
    t = np.array([pos[0], pos[1], pos[2]])

    # 쿼터니언 -> 회전 행렬 변환 (정규화된 쿼터니언을 가정)
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])
    
    # RT: extrinsic matrix (4x4) 구성
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = t

    # RT_inv: RT의 역행렬 (회전은 전치, translation은 -R^T * t)
    R_inv = R.T
    t_inv = - R_inv @ np.array(pos)
    RT_inv = np.eye(4)
    RT_inv[:3, :3] = R_inv
    RT_inv[:3, 3] = t_inv

    return RT, RT_inv


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




def recon_3d_frame(c_abs_pose, depth_map, K_inv, RT_inv, realsense=False, downscale=1):
    depth_map[depth_map > 4.5] = 0
    depth_map = torch.tensor(depth_map, device=DEVICE)
    
    pose4 = torch.cat((c_abs_pose, torch.tensor([0.0], device=DEVICE)))
    pose4 = pose4.view(1, 4)

    if realsense:
        pixel_full = torch.stack((
            TG_PIXEL_X_REAL[::downscale,::downscale].reshape((-1))*depth_map.reshape((-1)), 
            TG_PIXEL_Y_REAL[::downscale,::downscale].reshape((-1))*depth_map.reshape((-1)), 
            depth_map.reshape((-1))
            ))
    else:
        pixel_full = torch.stack((
            TG_PIXEL_X[::downscale,::downscale].reshape((-1))*depth_map.reshape((-1)), 
            TG_PIXEL_Y[::downscale,::downscale].reshape((-1))*depth_map.reshape((-1)), 
            depth_map.reshape((-1))
            ))
    intrinsic = torch.matmul(K_inv.float().to(DEVICE), pixel_full.float())
    intrinsic_homo = torch.cat((intrinsic, torch.ones(1, intrinsic.shape[1], device=DEVICE)), dim=0)  # (4×256)

    extrinsic = torch.matmul(RT_inv.float(), intrinsic_homo.float())

    extrinsic = extrinsic.T
    
    pose_matrix = pose4.repeat(extrinsic.size(0), 1)

    final = extrinsic
    comparison = (final == pose4)
    matches = ~torch.all(comparison, dim=1)
    matching_indices = torch.nonzero(matches).squeeze()

    return final, matching_indices

def build_pointcloud_rgb(result, color1):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(result.cpu().numpy())
    point_cloud.colors = o3d.utility.Vector3dVector((color1 / 255.0).cpu().numpy())

    mat_pc = o3d.visualization.rendering.MaterialRecord()
    mat_pc.shader = "defaultLitTransparency"
    mat_pc.base_color = [1,1,1,0.75]

    point_cloud_list = [{'name':'point_cloud_rgb','group' : 'point_cloud', 'geometry':point_cloud, 'material':mat_pc}]

    return point_cloud_list

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: 
        replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def base_pointcloud(extra_info_path, downscale = 4, realsense=False):
    _, K_inv = intrinsic_matrix_torch(realsense)
    frame_list = sorted(os.listdir(extra_info_path))
    
    result_full = torch.empty((0, 3), dtype=torch.float32, device=DEVICE)
    rgb_result = torch.empty((0, 3), dtype=torch.uint8, device=DEVICE)

    for frame_i, frame in tqdm(enumerate(frame_list), desc="Pointcloud"):

        rgb_path = os.path.join(extra_info_path, frame, 'original_image.png')
        depth_path = os.path.join(extra_info_path, frame, 'depth.npy')
        pose_ori_path = os.path.join(extra_info_path, frame, 'pose_ori.npy')

        rgb_image = cv2.imread(rgb_path) 
        rgb_image = rgb_image[::downscale, ::downscale]
        depth_map = torch.tensor(np.load(depth_path), device=DEVICE)
        depth_map = depth_map[::downscale, ::downscale]

        pose_ori = np.load(pose_ori_path, allow_pickle=True)
        c_abs_pose = torch.tensor(pose_ori[0], dtype=torch.float32, device=DEVICE)
        c_abs_ori = torch.tensor(pose_ori[1], dtype=torch.float32, device=DEVICE)

        _, RT_inv = extrinsic_matrix_torch(c_abs_ori, c_abs_pose)

        temp_result, sort = recon_3d_frame(c_abs_pose, depth_map, K_inv, RT_inv, realsense=realsense, downscale=downscale)
        result_full = torch.cat((result_full, (temp_result[sort])[:,:-1]), dim=0)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_result = torch.cat((rgb_result, torch.tensor(rgb_image, device=DEVICE).view(-1, 3)[sort]), dim=0)

    pointcloud = build_pointcloud_rgb(result=result_full, color1=rgb_result)

    return pointcloud


def save_txt(extra_info_path, output_txt_path):
    frame_paths = sorted(os.listdir(extra_info_path))
    fx, fy, cx, cy = get_intrinsic_parameters(realsense=False)
    camera_params = {
        "axisAlignment": "1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000",
        "colorHeight": 1024,
        "colorToDepthExtrinsics": "1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000",
        "colorWidth": 1024,
        "depthHeight": 1024,
        "depthWidth": 1024,
        "fx_color": f"{fx:.6f}",
        "fx_depth": f"{fx:.6f}",
        "fy_color": f"{fy:.6f}",
        "fy_depth": f"{fy:.6f}",
        "mx_color": f"{cx:.6f}",
        "mx_depth": f"{cx:.6f}",
        "my_color": f"{cy:.6f}",
        "my_depth": f"{cy:.6f}",
        "numColorFrames": len(frame_paths),
        "numDepthFrames": len(frame_paths),
        "numIMUmeasurements": 2460,
    }
    # TXT 파일로 저장
    with open(output_txt_path, 'w') as f:
        for key, value in camera_params.items():
            if isinstance(value, list):
                f.write(f"{key} =\n")
                for row in value:
                    f.write(" ".join(map(str, row)) + "\n")
            else:
                f.write(f"{key} = {value}\n")
    
    print(f"Camera parameters saved to {output_txt_path}")


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


def convert_rt_isaac_to_scannet(RT_isaac):
    """
    Convert RT matrix from Isaac Sim to ScanNet using PyTorch.

    Args:
        RT_isaac (torch.Tensor): 4x4 extrinsic matrix in Isaac Sim coordinates.

    Returns:
        torch.Tensor: 4x4 extrinsic matrix in ScanNet coordinates.
    """
    # Coordinate transformation matrix (as torch.Tensor)
    M = torch.tensor([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=RT_isaac.dtype, device=RT_isaac.device)

    # Extract R and T from RT matrix
    R_isaac = RT_isaac[:3, :3]  # Rotation (3x3)
    T_isaac = RT_isaac[:3, 3]   # Translation (3x1)

    # Convert rotation and translation
    # R_scannet = M @ R_isaac @ M.T
    R_scannet = R_isaac
    T_scannet = M @ T_isaac

    # Construct new RT matrix
    RT_scannet = torch.eye(4, dtype=RT_isaac.dtype, device=RT_isaac.device)
    RT_scannet[:3, :3] = R_scannet
    RT_scannet[:3, 3] = T_scannet

    return RT_scannet



def get_cam_pose(pose_ori_path, output_pose_path):
    pose_ori = np.load(pose_ori_path, allow_pickle=True)
    c_abs_pose = torch.tensor(pose_ori[0], dtype=torch.float32, device=DEVICE)
    c_abs_ori = torch.tensor(pose_ori[1], dtype=torch.float32, device=DEVICE)
    RotT, RotT_inv = extrinsic_matrix_torch(c_abs_ori, c_abs_pose)
    # RotT, RotT_inv = get_extrinsic(pose_ori[1], pose_ori[0])
    # RT_scannet = convert_rt_isaac_to_scannet(RT_inv)
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

DEPTH_TRUNC = 20

def backproject(color, depth, intrinisc_cam_parameters, extrinsics):
    """
    Convert RGB and depth images to a colored point cloud.
    """
    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinisc_cam_parameters
    )
    pcd.transform(extrinsics)
    return pcd

def create_point_cloud_from_frames(base_dir, stride=1, max_points_per_frame=10000, max_total_points=500000):
    depth_dir = os.path.join(base_dir, 'depth')
    color_dir = os.path.join(base_dir, 'color')
    pose_dir = os.path.join(base_dir, 'pose')
    intr_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')

    image_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('.')[0]))
    # end = int(image_list[-1].split('.')[0]) + 1
    # frame_id_list = np.arange(0, end, stride)
    # frame_ids = list(frame_id_list)
    frame_ids = [x.split('.')[0] for x in image_list][::stride]
    intrinisc_cam_parameters = get_intrinsics(image_size, intr_path)

    all_points = []
    all_colors = []

    with tqdm(total=len(frame_ids)) as pbar:
        for i, fid in enumerate(frame_ids):
            pbar.set_description(f"Pointcloud (Frame {fid})")
            # Your processing code here
            pbar.update(1)
            # print(f"Processing frame {fid}...")

            depth_path = os.path.join(depth_dir, f"{fid}.png")
            depth = get_depth(depth_path)
            color = imageio.v2.imread(os.path.join(color_dir, f"{fid}.jpg"))
            # Resize color image to match depth image shape
            if color.shape[:2] != depth.shape:
                color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            extrinsics = read_pose(os.path.join(pose_dir, f"{fid}.txt"))

            pcd = backproject(color, depth, intrinisc_cam_parameters, extrinsics)

            # ✨ Limit number of points per frame
            if len(pcd.points) > max_points_per_frame:
                idx = np.random.choice(len(pcd.points), max_points_per_frame, replace=False)
                pcd = pcd.select_by_index(idx)

            all_points.append(np.asarray(pcd.points))
            all_colors.append(np.asarray(pcd.colors))

    print("Concatenating point cloud...")
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        # ✨ 최종 point 수 500,000개로 제한
        if len(all_points) > max_total_points:
            idx = np.random.choice(len(all_points), max_total_points, replace=False)
            all_points = all_points[idx]
            all_colors = all_colors[idx]
    else:
        all_points = np.zeros((0, 3))
        all_colors = np.zeros((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    return pcd

def save_voxel_grid_as_colored_point_cloud(voxel_grid, output_path):
    voxels = voxel_grid.get_voxels()
    voxel_size = voxel_grid.voxel_size

    # 중심 좌표 계산: grid_index * voxel_size + origin
    voxel_centers = np.array([v.grid_index for v in voxels], dtype=np.float32)
    voxel_centers = voxel_centers * voxel_size + voxel_grid.origin

    # 색상 추출
    voxel_colors = np.array([v.color for v in voxels], dtype=np.float32)

    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

    # 저장
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Colored voxel grid saved as point cloud at: {output_path}")


if __name__=="__main__":
    scene_path = '/workspace/data/tasmap/capture_test/cook/2a77be2c-c48a-4fd1-b27e-c0153673f884/Kitchen-19107/Kitchen_cook/frames/extra_info'

    data_root_dir = os.path.join('/workspace/MaskClustering/data/tasmap', 'processed')
    scene_name = 'scene0000_00'

    data_dir = os.path.join(data_root_dir, scene_name)
    os.makedirs(data_dir, exist_ok=True)

    # # save 2D images
    # save_2D(scene_path, data_dir)

    # # save ply
    # output_ply_path = os.path.join(data_dir, f'{scene_name}_vh_clean_2.ply')
    # pcd = base_pointcloud(scene_path, downscale=32, realsense=False)
    # o3d.visualization.draw(pcd, show_skybox=False, bg_color=(0,0,0,1))
    # print("Saving pointcloud")
    # o3d.io.write_point_cloud(output_ply_path, pcd[0]['geometry'], write_ascii=False)

    # make ply
    output_ply_path = os.path.join(data_dir, f'{scene_name}_vh_clean_2.ply')
    image_size = (1024, 1024)
    # image_size = (640, 480)
    depth_scale = 1000
    max_points_per_frame=20000
    max_total_points=500000
    stride = 1

    pcd = create_point_cloud_from_frames(data_dir, stride=stride, max_points_per_frame=max_points_per_frame, max_total_points=max_total_points)

    # # save ply
    # o3d.io.write_point_cloud(output_ply_path, pcd, write_ascii=False)

    # save downsampled ply
    voxel_size = 0.005
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3d.visualization.draw_geometries([pcd_down], window_name="Downsampled Point Cloud")
    o3d.io.write_point_cloud(output_ply_path, pcd_down)


    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_grid], window_name="ScanNet Point Cloud")

    # voxel_output_path = os.path.join(data_dir, f'{scene_name}_voxel_colored.ply')
    # save_voxel_grid_as_colored_point_cloud(voxel_grid, voxel_output_path)