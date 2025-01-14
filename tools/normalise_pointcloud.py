import laspy

"""sets the starting x, y, z values of a las point cloud to 0 """


# Load the LAS file
input_path = r"data/Rolleston_lidar_20230707.las"
output_path = r"data/Rolleston_lidar_20230707_normalised.las"

# Read the file
las = laspy.read(input_path)

# Extract the minimum values for normalization
x_min, y_min, z_min = las.x.min(), las.y.min(), las.z.min()

# Normalize the coordinates
normalized_x = las.x - x_min
normalized_y = las.y - y_min
normalized_z = las.z - z_min

# Update the LAS header to adjust the offset
las.header.offsets = [0.0, 0.0, 0.0]  # Set the new offsets for X, Y, Z
las.header.scales = [0.01, 0.01, 0.01]  # Keep scales small to retain precision

# Update the coordinates in the LAS file
las.x = normalized_x
las.y = normalized_y
las.z = normalized_z

# Save the normalized file
las.write(output_path)

print(f"Point cloud normalized and saved to {output_path}.")
