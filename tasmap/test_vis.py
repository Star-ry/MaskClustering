import open3d as o3d
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
o3d.visualization.draw_geometries([mesh])
