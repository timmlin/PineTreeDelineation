import numpy as np
import scipy
import copy

# -------------------------------------------------
#                UTILITY FUNCTIONS
# -------------------------------------------------
def _dimension(points):
    """
    Takes a numpy array and returns a flag indicating the dimention of the array. 
    Will raise an error if the array is not suitable for use in the rest of the script.
    """
    if points.shape[1] == 3:
        dim = '3d'
    elif points.shape[1] == 2:
        dim = '2d'
    else:
        raise Exception('Provided points must be 2d points or 3d points.')
    return dim

def _delete(array, indices):
    """
    indices:  must be list of integers or numpy array of integers
    """
    if isinstance(indices, int):
        indices = [indices]
        
    if isinstance(array, np.ndarray):
        array_out = np.delete(array, indices, axis = 0)
        return array_out
    if isinstance(array, list):
        array =  np.array(copy.deepcopy(array))
        array_out = np.delete(array, indices, axis = 0)
        return array_out
        # sorted_indices = sorted(indices, reverse = True)
        # array_out = copy.deepcopy(array)
        # for i in sorted_indices:
        #     del array_out[i]
        # return array_out


def _encode(rows, n_rows: int, cols, n_cols: int, hits = None):
    """
    DESCRIPTION: A helper function for calculating sequential tags for a voxel space.

    INPUT
    rows:       numpy array of length n containing the voxel row address of each of the n input points
    n_rows:     integer indicating the number of rows in the voxel space
    cols:       numpy array of length n containing the voxel column address of each of the n input points
    n_cols:     integer indicating the number of columns in the voxel space
    hits:       (optional) numpy array of length n containing the voxel hight address of each of the n input points
    verbose:    flag which enables/supresses status printing

    OUTPUT
    A (nx1) numpy array contining the voxel cell tag for each of the n input points
    """
    if hits is not None:
        tags = hits*n_rows*n_cols + rows*n_cols + cols 
    else:
        tags = rows*n_cols + cols
    return tags

def _decode(tags, n_rows, n_cols, dim):
    """
    DESCRIPTION: A helper function which returns the voxel space address given a voxel tag.

    INPUT
    tags:       numpy array containing the tags to decode
    n_rows:     integer indicating the number of rows in the voxel space
    n_cols:     integer indicating the number of columns in the voxel space
    dim:        a flag indicating the dimentionality of the voxel space

    OUTPUT
    A (n x dim) numpy array containing the voxel space addresses of the tags
    """
    if dim == '3d':
        hits, rem = np.divmod(tags, n_rows*n_cols)
        rows, rem = np.divmod(rem, n_cols)
        cols = rem
        return np.column_stack((rows, cols, hits))
    else:
        rows, rem = np.divmod(tags, n_cols)
        cols = rem
        return np.column_stack((rows, cols))

def _voxelize(points, vox_size, verbose = False):
    """
    INPUT
    points:         A numpy array where rows represent points and columns contain X, Y, and Z values of each point.
    vox_size        A float value indicating the size of inidividual voxels
    verbose         (False) A flag enabling or supresing status printing

    OUTPUT
    point_vox_tags  A numpy array containing the superpoint tag ID of each point
    vox_centers     A numpy array containing the center coordinate of each voxel
    vox_tags        A numpy array containing the tag for each voxel
    """

    # Get dimensionality of the points variable
    dim = _dimension(points)

    # Calculate span of the point space (old: int(np.amin(points[:,0]) - 5*vox_size))
    nwt_x = np.amin(points[:,0]) - (vox_size/2) #north west top 
    nwt_y = np.amax(points[:,1]) + (vox_size/2)
    seb_x = np.amax(points[:,0]) + (vox_size/2) #south east bottom
    seb_y = np.amin(points[:,1]) - (vox_size/2)
    n_rows = int(np.ceil((nwt_y - seb_y)/vox_size)) #number of rows
    n_cols = int(np.ceil((seb_x - nwt_x)/vox_size))
    if dim == '3d':
        nwt_z = np.amax(points[:,2]) + (vox_size/2)

    # Calculate voxel row, col, (hight) for each point
    rows = np.int_(np.subtract(nwt_y, points[:,1])/vox_size)
    cols = np.int_(np.subtract(points[:,0], nwt_x)/vox_size)
    if dim == '3d':
        hits = np.int_(np.subtract(nwt_z, points[:,2])/vox_size)

    # Superpoint ID Calculation for each point
    if dim == '3d':
        point_vox_tags = _encode(rows, n_rows, cols, n_cols, hits)
    else:
        point_vox_tags = _encode(rows, n_rows, cols, n_cols)

    # Get unique names
    vox_tags = np.unique(point_vox_tags)
    vox_addresses = _decode(vox_tags, n_rows, n_cols, dim)
    
    x = nwt_x + vox_addresses[:,1]*vox_size + vox_size/2 #center x for all addresses
    y = nwt_y - vox_addresses[:,0]*vox_size - vox_size/2 #center y for all addresses
    if dim == '3d':
        z = nwt_z - vox_addresses[:,2]*vox_size - vox_size/2 #center z for all addresses
        vox_centers = np.column_stack((x,y,z))
    else:
        vox_centers = np.column_stack((x,y))

    return point_vox_tags, vox_centers, vox_tags


def _density(points, sp_point_indices, verbose = False):
    """
    INPUT
    points:             A numpy array where rows represent points and columns contain X, Y, and Z values of each point.
    sp_point_indices    A numpy array where each row contains the list indicies of all points belonging to each superpoint 
    verbose             (False) A flag enabling or supresing status printing

    OUTPUT
    sp_mass_centers     A numpy array containing the average xyz for each voxel shape
    sp_densities        A numpy array containing the number of points in each voxel shape
    """
    # Initialize
    sp_mass_centers   = []
    sp_densities = []

    # Get dimensionality of the points variable
    dim = _dimension(points)

    # Calculate Voxel Statistics
    for indices in sp_point_indices:

        # Calculate statistics
        pts = points[indices]
        cx = np.average(pts[:,0])
        cy = np.average(pts[:,1])
        if dim == '3d':
            cz = np.average(pts[:,2])
            sp_mass_centers.append([cx, cy, cz])
        else:
            sp_mass_centers.append([cx, cy])
        sp_densities.append(len(indices))

    # Wrap up
    sp_mass_centers = np.array(sp_mass_centers)
    return sp_mass_centers, sp_densities


# -------------------------------------------------
#            INITIALIZER MODIFIERS
# -------------------------------------------------
def modifier_initialize(constructor, sp_centers = None, sp_point_indices = None):
    if sp_centers is not None:
        constructor.set_sp_centers(sp_centers)
    if sp_point_indices is not None:
        constructor.set_sp_point_indices(sp_point_indices)


def modifier_initialize_by_points(constructor):
    '''
    DESCRIPTION: Initialize by the points already in the constructor
    '''
    constructor.set_sp_centers(constructor.get_points())
    constructor.set_sp_point_indices(np.arange(len(constructor.get_points())))


def modifier_initialize_by_centers_and_sphere(constructor, sp_centers = None, radius = 0, add_on = False):
    '''
    DESCRIPTION: Initializes the space based on provided centers and a radius. If no centers are provided,
                 existing centers will be used.
    '''
    if not sp_centers:
        try:
            sp_centers = constructor.get_sp_centers()
        except Exception:
            raise Exception('No sp_centers were provided to the constructor.')
    constructor.set_sp_centers(sp_centers, add_on = add_on)
    modifier_collect_by_sphere(constructor, radius, sp_centers = sp_centers, add_on = add_on)


def modifier_initialize_by_cube(constructor, size):
    '''
    DESCRIPTION: An INITIALIZER function which divides a point space into cubes.       
    '''
    # Get voxel superpoint_ids
    points = constructor.get_points()
    point_sp_tags, sp_centers, sp_tags  = _voxelize(points, size)

    # Create dictionary to organize indices by voxel
    temp_dict = {}       
    for i in range(len(points)):
        temp_dict.setdefault(point_sp_tags[i], []).append(i)

    # Break up dictionary into list of indices per voxel
    sp_point_indices = []
    for tag in sp_tags: #keeps order of indices in correspondance with sp_tags and sp_centers
        if tag not in temp_dict:
            raise Exception(f'Superpoint {tag} was not found to contain any point indices.')
        sp_point_indices.append(temp_dict[tag])

    constructor.set_sp_centers(sp_centers)
    constructor.set_sp_point_indices(sp_point_indices)


def modifier_initialize_by_sphere(constructor, radius):

    # Determine voxel size from radius. voxel should be circumscribed by sphere
    points = constructor.get_points()
    dim = _dimension(points)
    if dim =='3d': 
        vox_size = 2*radius/np.sqrt(3)
    #    self.sp_volume = (4/3)*np.pi*radius**3
    elif dim == '2d':
        vox_size = 2*radius/np.sqrt(2)
    #    self.sp_volume = np.pi*radius**2

    # Get voxel superpoint_ids
    _, sp_centers, sp_tags = _voxelize(points, vox_size)

    # Create list of list of indices per sphere
    if not hasattr(constructor, 'KDtree'):
        constructor.KDtree = scipy.spatial.cKDTree(points)

    sp_point_indices = constructor.KDtree.query_ball_point(sp_centers, radius)

    constructor.set_sp_centers(sp_centers)
    constructor.set_sp_point_indices(sp_point_indices)


# -------------------------------------------------
#            COLLECTOR MODIFIERS
# -------------------------------------------------
def modifier_collect_by_sphere(constructor, radius, sp_centers = None, add_on = False):
    if sp_centers is None:
        sp_centers = constructor.get_sp_centers()
    if not hasattr(constructor, 'KDtree'):
        constructor.KDtree = scipy.spatial.cKDTree(constructor.get_points())
    sp_point_indices = constructor.KDtree.query_ball_point(sp_centers, radius)
    constructor.set_sp_point_indices(sp_point_indices, add_on = add_on)


def modifier_collect_by_nearest(constructor, sp_centers = None, add_on = False):
    if sp_centers is None:
        sp_centers = constructor.get_sp_centers()
    sp_centerKDTree = scipy.spatial.cKDTree(constructor.get_sp_centers())
    _, indices = sp_centerKDTree.query(constructor.get_points()) #<---- gives indices of sp_centers for each point. need to reverse

    # flip indices
    temp_dict = {}
    for point_i, sp_center_i in enumerate(indices):
        temp_dict.setdefault(sp_center_i, []).append(point_i)
        
    sp_point_indices = []
    for i in range(len(sp_centers)):
        if i in temp_dict:
            sp_point_indices.append(temp_dict[i])
        else:
            sp_point_indices.append([])
            
    constructor.set_sp_point_indices(sp_point_indices, add_on = add_on)
    modifier_center_filter_zeros(constructor)


# -------------------------------------------------
#              CENTERING MODIFIERS
# -------------------------------------------------
def modifier_center_by_com(constructor):
    sp_mass_centers, _ = _density(constructor.get_points(), constructor.get_sp_point_indices())
    constructor.set_sp_centers(sp_mass_centers)


def modifier_center_by_nearest(constructor):
    if not hasattr(constructor, 'KDtree'):
        constructor.KDtree = scipy.spatial.cKDTree(constructor.get_points())
    _, ind = constructor.KDtree.query(constructor.get_sp_centers(), k = 1)
    sp_mass_centers = constructor.get_points()[ind,:] #replace average centers with actual point as center
    
    constructor.set_sp_centers(sp_mass_centers)


def modifier_center_filter_zeros(constructor):
    sp_point_indices = constructor.get_sp_point_indices()
    to_delete = []
    for i, indices in enumerate(sp_point_indices):
        if len(indices) == 0:
            to_delete.append(i)
    constructor.delete_sp_index(to_delete)


def modifier_center_filter_by_feature(constructor, feature_key, threshold, equality = 'lt'):
    '''
    Does not allow filtering by sub dictionary. Will fail if feature key maps to dict of dict
    '''
    feature = constructor.get_sp_feature(feature_key)
    if isinstance(feature, list):
        feature = np.array(feature)

    if equality == 'lt':
        to_delete = list(np.where(feature < threshold)[0])
    elif equality == 'gt':
        to_delete = list(np.where(feature > threshold)[0])
    else:
        raise Exception('passed equality is not supported')

    constructor.delete_sp_index(to_delete)


# -------------------------------------------------
#                 CALCULATORS
# -------------------------------------------------
def modifier_calculate_density(constructor, key = 'den'):
    _, density = _density(constructor.get_points(), constructor.get_sp_point_indices())
    constructor.set_sp_feature(key, density)


def modifier_calculate_com(constructor, key = 'com'):
    center_of_mass, _ = _density(constructor.get_points(), constructor.get_sp_point_indices())
    constructor.set_sp_feature(key, center_of_mass)


def modifier_calculate_eigenstuff(constructor, lmdkey = 'lmb', veckey = 'vec', shpkey = 'shp'):
    lmd, vec, shp = _calculate_eigenstuff(constructor.get_points(), constructor.get_sp_point_indices())
    constructor.set_sp_feature(lmdkey, lmd)
    constructor.set_sp_feature(veckey, vec)
    constructor.set_sp_feature(shpkey, shp)
        
        
def modifier_set_feature(constructor, key, values):
    constructor.set_sp_feature(key, values)