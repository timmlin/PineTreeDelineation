import laspy
import numpy as np
import open3d as o3d



def visualise_las(filepath):
    las = laspy.read(filepath)

    # Extract points
    points = np.vstack((las.x, las.y, las.z)).T

    # Create an Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    

def visualise_las_in_sections(filepath, chunk_size):
    # Read LAS file
    las = laspy.read(filepath)
    
    # Extract point coordinates
    points = np.vstack((las.x, las.y, las.z)).T
    total_points = points.shape[0]
    
    # Process in chunks
    for i in range(0, total_points, chunk_size):
        end_index = min(i + chunk_size, total_points)
        chunk_points = points[i:end_index]

        # Create Open3D PointCloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(chunk_points)

        # Visualize the downsampled point cloud
        print(f"Processing points {i} to {end_index}")
        o3d.visualization.draw_geometries([point_cloud])

        # # Optionally, save the downsampled chunk
        # output_file = f"chunk_{i}.ply"
        # o3d.io.write_point_cloud(output_file, downsampled_pcd)
        # print(f"Saved chunk to {output_file}")



def split_las(filepath):

    las = laspy.read(filepath)

    # Extract points
    points = np.vstack((las.x, las.y, las.z)).T

    # Create an Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)



    grid_boundaries = [
        (np.array([0, 0, 0]), np.array([10, 10, 10])),
        (np.array([10, 0, 0]), np.array([20, 10, 10])),
        (np.array([20, 10, 0]), np.array([30, 20, 10])),
        # Add more regions as needed
    ]

    # Create lines for visualizing boundaries
    line_sets = []
    for min_bound, max_bound in grid_boundaries:
        # Corners of the bounding box
        corners = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ]
        corners = np.array(corners)

        # Edges of the bounding box (pairs of indices into the corners array)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]

        # Create the LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in edges])  # Red lines

        line_sets.append(line_set)

    o3d.visualization.draw_geometries([point_cloud, *line_sets])





# File path
las_file_path = r"PineTreeDelineation/data/UAV_sample_data/plot_31_pointcloud.las"


split_las(las_file_path)