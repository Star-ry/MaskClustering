import numpy as np
import open3d as o3d
import open3d.visualization as vis
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.config import get_dataset, get_args

# Set seed for consistent color generation
np.random.seed(6)

def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    color = (np.random.rand(3) * 0.7 + 0.3)
    colors = np.tile(color, (points.shape[0], 1))
    return point_ids, points, colors, color, np.mean(points, axis=0)


def main(args):
    point_size = 5.0
    label_colors, labels, centers = [], [], []

    dataset = get_dataset(args)
    mesh = o3d.io.read_triangle_mesh(dataset.mesh_path)
    scene_points = np.asarray(mesh.vertices)
    scene_points = scene_points - np.mean(scene_points, axis=0)
    scene_colors = np.asarray(mesh.vertex_colors)
    scene_colors = np.power(scene_colors, 1/2.2)  # Brighten colors
    scene_colors = np.clip(scene_colors, 0, 1)

    pred = np.load(f'/workspace/MaskClustering/data/prediction/{args.config}_class_agnostic/{args.seq_name}.npz')
    masks = pred['pred_masks']
    num_instances = masks.shape[1]

    instances_list = []
    instance_colors = np.zeros_like(scene_colors)

    for idx in range(num_instances):
        mask = masks[:, idx]
        point_ids = np.where(mask)[0]

        point_ids, points, colors, label_color, center = vis_one_object(point_ids, scene_points)
        instance_colors[point_ids] = label_color
        label_colors.append(label_color)
        labels.append(str(idx))
        centers.append(center)

        # Add individual instance point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        instance = [{'name':f'obj_{idx}','group' : 'point_cloud', 'geometry':pcd}]

        instances_list += instance

    # RGB full point cloud (optional)
    full_scene = o3d.geometry.PointCloud()
    full_scene.points = o3d.utility.Vector3dVector(scene_points)
    full_scene.colors = o3d.utility.Vector3dVector(scene_colors)
    full_scene_list = [{'name':'RGB','group' : 'point_cloud', 'geometry':full_scene}]

    # Combined instance-colored point cloud
    labeled_mask = np.sum(instance_colors, axis=1) != 0
    labeled_scene = o3d.geometry.PointCloud()
    labeled_scene.points = o3d.utility.Vector3dVector(scene_points[labeled_mask])
    labeled_scene.colors = o3d.utility.Vector3dVector(instance_colors[labeled_mask])
    labeled_scene_list = [{'name':'labeled_scene','group' : 'point_cloud', 'geometry':labeled_scene}]

    # Draw the scene
    # o3d.visualization.draw_geometries(all_points, point_show_normal=False, width=1280, height=720)
    vis.draw(instances_list+full_scene_list+labeled_scene_list, show_skybox=False, bg_color=(0,0,0,1))


if __name__ == '__main__':
    args = get_args()
    depth_trunc=0.005
    main(args)
