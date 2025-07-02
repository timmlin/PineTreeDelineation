import numpy as np
import matplotlib.pyplot as plt
import laspy
import scipy
import open3d as o3d
import time
import sys


from ofmp import superpoint_construction as sp
from ofmp import superpoint_methods as sm
from ofmp import networks as nw

from tools.utils import *

    
# User Parameters
s = 0.15     # Voxel grid size
g_max = 1   # maximum ground height after flattening
c_min = 16   # shortest tree to detect
d = 0.01     # Merge radius
k = 30      # number of connections between nodes


files = []
directory = "data/rolleston_forest_plots"
for filename in os.listdir(directory):
        if filename.startswith("plot") and filename.endswith(".las"):
            file_path = os.path.join(directory, filename)
            # Perform operations on the file here
            files.append(file_path)

#Run Script on every las file in dir
for filename in files[1:]:

    print(filename)

    start_time = time.time()

    las = laspy.read(filename)

    las = noramlise_las(las)
    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld = pnt_cld.voxel_down_sample(voxel_size=0.1)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    #Convert back to np array
    points = np.asarray(pnt_cld.points)

    points = classify_ground_threshold(points, 1, visualise = False)



    tree_points, ground_points = points

    # Create the superpoint space design
    design = sp.SuperpointSpaceDesign()
    layer1 = design.add_design_layer(layer_name = 'layer1')
    layer1.add_modifier(sm.modifier_initialize_by_cube, size = s) # voxelizes and sets centers
    layer1.add_modifier(sm.modifier_center_by_com)                  # moves centers to center of mass
    layer1.add_modifier(sm.modifier_center_by_nearest)              # snaps center to nearest point

    # Build the space
    space_constructor = sp.SuperpointSpaceConstructor(design, tree_points)
    space = space_constructor.build_space()
    network  = nw.create_network_from_superpointspace_knn(space = space, k = k, weight_method = nw.sqr_dist)




    # ID ground superpoints and contract network
    cents    = space.sp_centers
    KDtree   = scipy.spatial.cKDTree(ground_points[:,0:2]) #build KDtree from dtm points
    dist,ind = KDtree.query(cents[:,0:2], 10) #query tree with lidar points
    grnd_elv = ground_points[ind, 2]
    grnd_wts = 1/(dist+1)
    terrain_at_centers = np.sum(grnd_elv*grnd_wts, axis = 1)/np.sum(grnd_wts, axis=1)
    grnd_keys = np.where((cents[:,2]-terrain_at_centers) < g_max)[0].tolist() # <-----------GROUND HEIGHT
    nw.contract_nodes_to_node(network, grnd_keys)




    # Calculate shortest Path
    shortest_tree = nw.pathing_single_source_all_target(network, grnd_keys[0])
    forest = nw.split_into_trees(shortest_tree, grnd_keys[0])




    # Gather points based on networks and filter by height
    tree_sp_indices = {}
    for key, tree in forest.items():
        indices = list(tree.nodes)
        coords = space.sp_centers[indices]
        ht_min = min(coords[:,2])
        ht_max = max(coords[:,2])
        if (ht_max - ht_min) > c_min: #<---- HEIGHT FILTER
            trunk_pos = np.mean(coords[(coords[:,2]-ht_min) < c_min], axis=0)
            tree_sp_indices[key] = {'pos':trunk_pos, 'sp':indices}



    unvisited = dict(tree_sp_indices)  # copy so we can safely pop
    clusters = []

    while unvisited:
        key, data = unvisited.popitem()
        cluster_keys = [key]
        cluster_sp = list(data['sp'])
        cluster_positions = [data['pos']]

        keys_to_check = list(unvisited.keys())
        for other_key in keys_to_check:
            other_pos = unvisited[other_key]['pos']
            dist = np.linalg.norm(data['pos'] - other_pos)
            if dist < d:
                cluster_keys.append(other_key)
                cluster_sp.extend(unvisited.pop(other_key)['sp'])
                cluster_positions.append(other_pos)

        avg_pos = np.mean(cluster_positions, axis=0)
        clusters.append({'sp': cluster_sp, 'pos': avg_pos})



    tree_number = 1
    classification = np.zeros(len(tree_points))
    for i, cluster in enumerate(clusters):
        sp_indices = cluster['sp']
        pt_indices = space.get_point_indices(sp_indices)
        classification[pt_indices] = tree_number
        tree_number += 1


    prev_out = sys.stdout
    with open('output.txt', 'w') as f:
        sys.stdout = f

        las_summary(filename)
        print('\n', tree_number)
        print(f'{(time.time() - start_time) } seconds \n')
        print('\n' * 5)
        sys.stdout = prev_out


    visualise_segments = []
    unique_labels = np.unique(classification)
    total_trees = 0
    for label in unique_labels:
        if label == 0:
            continue  # label 0 is unclassified or background

        # Get points belonging to this tree
        indices = np.where(classification == label)[0]
        tree = tree_points[indices]

        # Skip empty trees
        if len(tree) == 0:
            continue

        # Create and color point cloud
        color = np.random.rand(3)
        tree_cloud = o3d.geometry.PointCloud()
        tree_cloud.points = o3d.utility.Vector3dVector(tree)
        tree_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(tree), 1)))

        visualise_segments.append(tree_cloud)
        total_trees += 1

    las
    # ground_cloud = o3d.geometry.PointCloud()
    # ground_cloud.points = o3d.utility.Vector3dVector(ground_points)
    # ground_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(ground_points))  # Red color
    # visualise_segments.append(ground_cloud)

    # # Visualize all segments
    # o3d.visualization.draw_geometries(visualise_segments, window_name="Segmentation")



    # # Write out classified tls data
    # las_out = '4D_7_2_03b_trans2_classed.las' 
    # outfile = laspy.LasData(header = tls.header)
    # outfile.add_extra_dim(laspy.ExtraBytesParams(
    #     name="tree_number",
    #     type=np.uint64,
    #     description="tree_number"
    # ))
    # outfile.x     = pts[:,0]
    # outfile.y     = pts[:,1]
    # outfile.z     = pts[:,2]
    # outfile.tree_number = classification
    # outfile.write(las_out)


