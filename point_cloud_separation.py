import laspy
import numpy as np
import open3d as o3d

# File path
file_path = r"data\Rolleston_lidar_20220714.las"

# Open the LAS file
with laspy.open(file_path) as plot:
    total_points = plot.header.point_count
    chunk_size = 10_000_000  # Read 10 million points at a time

    print(f"Total points: {total_points}")

    # Read the entire data into memory (efficient for chunk processing)
    points = plot.read()

    # Loop through in chunks
    for i in range(0, total_points, chunk_size):
        # Extract the current chunk of points
        end_index = min(i + chunk_size, total_points)
        chunk_points = points[i:end_index]

        # Convert chunk to numpy array (x, y, z)
        points_np = np.vstack((chunk_points.x, chunk_points.y, chunk_points.z)).T

        # Create Open3D PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)

        # Downsample the point cloud for visualization
        voxel_size = 1.0  # Adjust voxel size based on your data
        downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)

        # Visualize the downsampled point cloud
        print(f"Processing points {i} to {end_index}")
        o3d.visualization.draw_geometries([downsampled_pcd])

        # # Optionally, save the downsampled chunk
        # output_file = f"chunk_{i}.ply"
        # o3d.io.write_point_cloud(output_file, downsampled_pcd)
        # print(f"Saved chunk to {output_file}")

