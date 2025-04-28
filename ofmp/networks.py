import networkx as nx
import numpy as np
import scipy
import os
from . import superpoint_construction as sp
from . import superpoint_methods as sm


# -------------------------------------------------
#                UTILITY FUNCTIONS
# -------------------------------------------------
def _knn(k, points, query_points = None):
    if query_points is None:
        query_points = points
    KDtree = scipy.spatial.cKDTree(points) #build KDtree from all points
    dis, ind = KDtree.query(query_points, k+1) #only query tree with new points
    ind = ind[:,1:] #[:,0:] is the query point itself, so skip first index
    dis = dis[:,1:]
    return ind, dis

def _ball(radius, points, query_points = None):
    if query_points is None:
        query_points = points
    KDtree = scipy.spatial.cKDTree(points) #build KDtree from all points
    ind = KDtree.query_ball_point(query_points, radius) #only query tree with new points
    return ind

def _get_nearest_node(net, coord):
    node_coords = np.array([net.space.sp_centers[key] for key in net])
    distances = np.linalg.norm(node_coords - coord, axis=1)
    nearest_node_index = np.argmin(distances)
    nearest_node_key = list(net)[nearest_node_index]
    return nearest_node_key

def _is_integer(obj):
    if isinstance(obj, (int, np.int64)):
        return True
    else:
        return False
    
def _iterative_add(key, neys, network, weight_method = None):
    if weight_method is None:
        weight_method = network.weight_method
    network.add_node(key)
    
    if not hasattr(neys, '__iter__'):
        neys = [neys]
    
    for ney in neys:
        if ney == key: continue
        if weight_method:
            fcost = weight_method(key, ney, network.space)
            bcost = weight_method(ney, key, network.space)
        else:
            fcost = 0
            bcost = 0
        network.add_edge(key, ney, weight = fcost)
        network.add_edge(ney, key, weight = bcost)


# -------------------------------------------------
#                WEIGHT METHODS
# -------------------------------------------------
def dist(a, b, space):
    axyz = space.sp_centers[a]
    bxyz = space.sp_centers[b]
    return np.linalg.norm(axyz - bxyz)

def sqr_dist(a, b, space):
    axyz = space.sp_centers[a]
    bxyz = space.sp_centers[b]
    return np.linalg.norm(axyz - bxyz)**2

def lamb_sqr_dist(a, b, space):
    pnt = space.sp_centers[a]
    nei = space.sp_centers[b]
    
    dist = np.linalg.norm(pnt - nei)
    
    if space.layer1.den[a] < 2 or space.layer1.den[b] < 2 or dist == 0:
        lamb = 1
    else:
        pnt_dir = space.layer1.vec[a][:,0]
        nei_dir = space.layer1.vec[b][:,0]

        pnt2nei_dir = (pnt-nei)/np.linalg.norm(pnt-nei)

        alpha = abs(np.dot(pnt_dir, pnt2nei_dir))
        beta  = abs(np.dot(nei_dir, pnt2nei_dir))

        lamb = (1 - min(abs(alpha), abs(beta)))/2        
    
    weight = dist**2 * lamb
    return weight



# -------------------------------------------------
#             NETWORK OPERATORS
# -------------------------------------------------
def create_network_from_superpointspace_knn(space, k, weight_method = None):
    network  = DiNet(space, weight_method)
    ind, _   = _knn(k, space.sp_centers)
    indices  = space.sp_names
    for key, neys in zip(indices, ind):
        _iterative_add(key, neys, network = network, weight_method = weight_method)
    return network


def add_edges_by_knn(network, nodes, k, weight_method = None):
    space  = network.space
    ind, _ = _knn(k, space.sp_centers[network.nodes], space.sp_centers[nodes])
    for key, neys in zip(nodes, ind):
        _iterative_add(key, neys, network = network, weight_method = weight_method)


def create_network_from_superpointspace_sphere(space, radius, weight_method = None):
    network = DiNet(space, weight_method)
    ind     = _ball(radius, space.sp_centers)
    indices = space.sp_names
    mark = 0
    for key, neys in zip(indices, ind):
        mark += 1
        #print(f'{mark}/{len(indices)}', end = '\r')
        _iterative_add(key, neys, network = network, weight_method = weight_method)
    return network


def add_edges_by_sphere(network, nodes, radius, weight_method = None):
    space = network.space
    ind = _ball(radius, space.sp_centers[network.nodes], query_points = space.sp_centers[nodes])
    for key, neys in zip(nodes, ind):
        _iterative_add(key, neys, network = network, weight_method = weight_method)


def create_network_from_overlap(space, layer_names, weight_method = None):
    #get connections based on overlap
    point_sp_map = space.get_point_sp_map(layer_names) #point indices to sp_indices
    connections_map = {}
    for point_i, sp_indices in point_sp_map.items(): #sp_indices that include same point
        for sp_index in sp_indices:
            connections_map.setdefault(sp_index, []).extend(sp_indices)
    for key, value in connections_map.items(): #remove duplicates
        connections_map[key] = list(set(value))

    network = DiNet(space)
    for key, neys in connections_map.items():
        _iterative_add(key, neys, network = network, weight_method = weight_method)
    return network


def contract_nodes_to_node(net, nodes_to_contract, super_key = None, weight_method = None):
    '''
    Combines all nodes in <nodes_to_contract> into a single node identified by <super_key>.
    '''
    if not super_key:
        super_key = nodes_to_contract[0]

    contract_dict = {key: False for key in net.nodes}
    contract_dict.update({key: True for key in nodes_to_contract})

    # Create a new node for the super-node and add its edges
    net.add_node(super_key)
    for key in nodes_to_contract:
        if key == super_key: continue
        neys = list(net.neighbors(key))
        for ney in neys:
            if not contract_dict[ney] and ney != super_key:
                if not net.has_edge(super_key, ney):
                    if weight_method:
                        fweight = weight_method(key, ney, net.space)
                        bweight = weight_method(ney, key, net.space)
                    elif weight_method == 0:
                        fweight = 0
                        bweight = 0
                    else:
                        fweight = net[key][ney]['weight']
                        bweight = net[ney][key]['weight']
                    net.add_di_edge(super_key, ney, {'weight':fweight}, {'weight':bweight})
        # Remove the node to contract
        net.remove_node(key)


def deweight_nodes(net, nodes_to_deweight):
    nodes_to_deweight = set(nodes_to_deweight)
    for key in nodes_to_deweight:
        neys = list(net.neighbors(key))
        for ney in neys:
            if ney in nodes_to_deweight:
                fweight = 0
                bweight = 0
                net.add_di_edge(key, ney, {'weight':fweight}, {'weight':bweight})

def deweight_join_nodes(net, nodes_to_deweight):
    nodes_to_deweight = set(nodes_to_deweight)
    for key in nodes_to_deweight:
        for ney in nodes_to_deweight:
            fweight = 0
            bweight = 0
            net.add_di_edge(key, ney, {'weight':fweight}, {'weight':bweight})


def create_network_from_networkx_graph(graph, node_center_key:str):

    #collect positions and associate each node name with an index number
    name_index_dict = {}
    centers = []
    for name, i in zip(graph.nodes, np.arange(len(graph.nodes))):
        centers.append(graph.nodes[name][node_center_key])
        name_index_dict[name] = i
    centers = np.array(centers)

    # Create Superpoint Space
    design = sp.SuperpointSpaceDesign()
    layer1 = design.add_design_layer(layer_name = 'layer1')
    layer1.add_modifier(sm.modifier_initialize_by_point, sp_centers = centers) # voxelizes and sets centers
    space_constructor = sp.SuperpointSpaceConstructor(design, centers)
    space = space_constructor.build_space()

    network = DiNet(space)
    for key, ney in graph.edges:
        network.add_edge(key, ney, weight = None)
        network.add_edge(ney, key, weight = None)
    return network


def compose(path_net, net):
    net.add_nodes_from(path_net.nodes(data = True))
    net.add_edges_from(path_net.edges(data = True))


def extract_path(net, key_list):
    new_net = DiNet(net.space)
    for key in key_list:
        new_net.add_nodes_from([(key, net.nodes[key])]) # key , attrs
    for key, ney in zip(key_list[:-1], key_list[1:]):
        new_net.add_edges_from([(key, ney, net.edges[(key, ney)])])
        new_net.add_edges_from([(ney, key, net.edges[(ney, key)])])
    return new_net


def pathing_single_source_all_target(net, source):
    if source not in net or not _is_integer(source):
        source = _get_nearest_node(net, source)

    all_paths = nx.shortest_path(net, source, weight='weight')

    # Create an empty graph of the same type
    #out_net = nx.DiGraph() if net.is_directed() else nx.Graph()
    out_net = DiNet(net.space)

    # Extract paths efficiently
    for key, path in all_paths.items():
        for u, v in zip(path, path[1:]):  # Consecutive pairs as edges
            if net.has_edge(u, v):  # Ensure edge exists in original network
                weight = net[u][v].get('weight', 1)  # Preserve weight if present
                out_net.add_edge(u, v, weight=weight)

    return out_net


def split_into_trees(tree_net, split_node):
    # Get all neighbors (both predecessors and successors) of the split node
    attached_nodes = set(tree_net.predecessors(split_node)).union(set(tree_net.successors(split_node)))

    # Copy the graph and remove the split node
    G_split = tree_net.copy()
    G_split.remove_node(split_node)

    # Get weakly connected components
    components = list(nx.weakly_connected_components(G_split))

    connected_map = {}
    for component in components:
        connection = attached_nodes.intersection(component)
        key_node = list(connection)[0]
        subgraph = tree_net.subgraph(component).copy()
        connected_map[key_node] = convert_to_dinet(subgraph, tree_net)
    return connected_map


def convert_to_dinet(g: nx.DiGraph, template = None, space=None, weight_method=None):
    if template:
        space = template.space
        weight_method = template.weight_method
    dinet = DiNet(superpointspace=space, weight_method=weight_method)
    dinet.add_nodes_from(g.nodes(data=True))
    dinet.add_edges_from(g.edges(data=True))
    return dinet


# -------------------------------------------------
#             NETWORK CLASSES
# -------------------------------------------------
class DiNet(nx.DiGraph):
    def __init__(self, superpointspace = None, weight_method = None):
        super().__init__()
        self.space = superpointspace
        self.weight_method = weight_method

    def attach_space(self, space):
        self.space = space

    def add_di_edge(self, u_of_edge, v_of_edge, fattr = None, battr = None):
        #TODO: usage should be more like super().add_edge(), ie: add_edge(u, v, weight = x)
        if not fattr and not battr and self.weight_method:
            fattr = {'weight':self.weight_method(u_of_edge, v_of_edge, self.space)}
            battr = {'weight':self.weight_method(v_of_edge, u_of_edge, self.space)}
            super().add_edge(u_of_edge, v_of_edge, **fattr)
            super().add_edge(v_of_edge, u_of_edge, **battr)
        elif not fattr and not battr:
            super().add_edge(u_of_edge, v_of_edge)
            super().add_edge(v_of_edge, u_of_edge)
        else:
            super().add_edge(u_of_edge, v_of_edge, **fattr)
            super().add_edge(v_of_edge, u_of_edge, **battr)

    def get_path_plotables(self, sp_list):
        coord = self.space.sp_centers
        dim = np.shape(coord)[1]
        xs = list(coord[sp_list,0])
        ys = list(coord[sp_list,1])
        if dim == 3:
            zs = list(coord[sp_list,2])
        elif dim == 2:
            zs = [0]*len(xs)
        return [xs, ys, zs]

    def get_plotables(self):
        lines  = []
        coords = self.space.sp_centers
        labels = self.space.sp_names
        for edge in self.edges:
            key, ney = edge
            xs, ys, zs = self.get_path_plotables([key, ney])
            lines.append([xs, ys, zs])
        return lines, labels, coords

    def write_polylines(self, fn):
        directory = os.path.dirname(fn)
        filename  = os.path.basename(fn)
        if not os.path.exists(directory):
            raise Exception('Directory does not exist.')
        if filename[-5:] != '.poly':
            filename = filename + '.poly'
        file = open('/'.join((directory, filename)),'w')

        printable = self.get_plotables()[0]
        for branch in printable:
            line = ''
            X = branch[0]
            Y = branch[1]
            Z = branch[2]
            for x,y,z in zip(X,Y,Z):
                line = line + '\t'.join([str(x),str(y),str(z)]) + '\n'
            file.write(line + '\n')
        file.close()

    def get_points_along_paths(self, spacing, random_range = 0):
        printable = self.get_plotables()[0]
        if spacing <= 0:
            raise Exception('<spacing> cannot be 0 or less')
        
        points = []
        mark = 0
        for branch in printable:
            mark += 1
            print(f'{mark}/{len(printable)}', end = '\r')
            point1 = np.array((branch[0][0],branch[1][0],branch[2][0]))
            point2 = np.array((branch[0][1],branch[1][1],branch[2][1]))
            num_points = int(np.linalg.norm(point1 - point2)/spacing) + 1

            # Create an array of parameter values between 0 and 1
            t_values = np.linspace(0, 1, num_points)
            
            # Calculate the coordinates of the points along the line using interpolation
            interpolated_points = np.outer(1 - t_values, point1) + np.outer(t_values, point2)
            
            points = points + interpolated_points.tolist()

        points = np.array(points)

        if random_range > 0:
            noise = np.random.uniform(-random_range, random_range, size = points.shape)
            points = points + noise

        return points
    
    def get_path_length(self, path, length_method = None):
        '''
        path is a list of nodes
        '''
        path_length = 0
        for u, v in zip(path[:-1], path[1:]):
            if length_method is None:
                path_length += self[u][v]['weight']
            else:
                path_length += length_method(u, v, self.space)
        return path_length