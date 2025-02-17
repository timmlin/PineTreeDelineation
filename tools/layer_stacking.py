import numpy as np
import laspy
import open3d as o3d


las = laspy.read("data/UAV_sample_data/plot_31_pointcloud.las")
points = np.vstack((las.x, las.y, las.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])

