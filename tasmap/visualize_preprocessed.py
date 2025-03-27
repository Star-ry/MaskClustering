import os
import numpy as np
import open3d as o3d
import imageio
import cv2


DEPTH_TRUNC = 20

def read_pose(path):
    return np.loadtxt(path)

def read_intrinsics(path):
    return np.loadtxt(path)


def get_intrinsics(image_size, intrinsic_path):
    intrinsics = np.loadtxt(intrinsic_path)

    intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
    intrinisc_cam_parameters.set_intrinsics(image_size[0], image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    return intrinisc_cam_parameters


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

def get_depth(depth_path):
    depth = cv2.imread(depth_path, -1)
    depth = depth / depth_scale
    depth = depth.astype(np.float32)
    return depth


def create_point_cloud_from_frames(base_dir, stride=1, max_points_per_frame=200000):
    depth_dir = os.path.join(base_dir, 'depth')
    color_dir = os.path.join(base_dir, 'color')
    pose_dir = os.path.join(base_dir, 'pose')
    intr_path = os.path.join(base_dir, 'intrinsic', 'intrinsic_depth.txt')

    image_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('.')[0]))
    # end = int(image_list[-1].split('.')[0]) + 1
    # frame_id_list = np.arange(0, end, stride)
    # frame_ids = list(frame_id_list)
    frame_ids = [x.split('.')[0] for x in image_list][::stride]
    intr = read_intrinsics(intr_path)
    intrinisc_cam_parameters = get_intrinsics(image_size, intr_path)

    all_points = []
    all_colors = []

    for i, fid in enumerate(frame_ids):
        # if i % stride != 0:
        #     continue
        print(f"Processing frame {fid}...")

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
    else:
        all_points = np.zeros((0, 3))
        all_colors = np.zeros((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    return pcd



if __name__ == "__main__":
    input_path = "/workspace/MaskClustering/data/tasmap/processed/scene0000_00"  # replace this
    output_ply = "scene_reconstructed.ply"
    image_size = (1024, 1024)
    # image_size = (640, 480)
    depth_scale = 1000
    max_points_per_frame = 100000
    stride=10

    pcd = create_point_cloud_from_frames(input_path, stride=stride, max_points_per_frame=max_points_per_frame)
    # Voxelization
    voxel_size = 0.05  # Adjust voxel size (smaller = more detail)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid], window_name="ScanNet Point Cloud")

    # print(f"Saving point cloud to {output_ply}")
    # o3d.io.write_point_cloud(output_ply, pcd)
