from . import superpoint_methods as sm
import numpy as np
import scipy

# Objective
# ------------------
# 1. Create a SuperpointSpace containing SuperpointLayer. 
# 2. Separate creation and use of the superpoint space object.
# 3. Multiple regions allowed around superpoint centers.
# 4. Allow user defined function to build superpoint space.

#TODO in the constructor, there are methods for getting space info. Make these SuperpointSpace methods and create method for subsetting space after creation

def _ensure_list(variable):
    if isinstance(variable, str):
        return [variable]
    try:
        iter(variable)
        return list(variable)
    except TypeError:
        return [variable]


# --------------------------------------
#            PRODUCT CLASSES
# --------------------------------------
class KeyDict(dict): #special dictionary where keys can be referenced like attributes
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class SuperpointSpace():
    def __init__(self, points):
        self.points           = points
        self.sp_centers       = None
        self.layers           = KeyDict()
        self.KDtree           = None

    @property
    def sp_names(self):
        return np.arange(len(self.sp_centers))

    def __len__(self):
        if self.sp_centers is not None:
            return len(self.sp_centers)
        else:
            return 0
        
    def __getattr__(self, name): #allows shorthand reference to specific layer
        if name in self.layers:
            return self.layers[name]
        else:
            return object.__getattribute__(self, name)
        
    def __getitem__(self, index):
        sp_dict = KeyDict()
        sp_dict['center'] = self.sp_centers[index]
        for name, layer in self.layers.items():
            sp_dict[name] = KeyDict()
            sp_dict[name]['point_indices'] = layer.sp_point_indices[index]
            for feature, value in layer.sp_features.items():
                sp_dict[name][feature] = value[index]
        return sp_dict
    
    def get_nearest_sp(self, points):
        if self.KDtree is None:
            self.KDtree = scipy.spatial.cKDTree(self.sp_centers)
        dis, ind = self.KDtree.query(points, 1)
        return dis, ind

    def get_point_sp_map(self, layer_names):
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        point_sp_map = {}
        for layer in layer_names:
            sp_point_indices = self.layers[layer].sp_point_indices
            for sp_i, indices in enumerate(sp_point_indices):
                for point_i in indices:
                    point_sp_map.setdefault(point_i, []).append(sp_i)
                    
        # Remove duplicates using set
        for key, value in point_sp_map.items():
            point_sp_map[key] = list(set(value))

        return point_sp_map
    
    def get_point_indices(self, keys = None, layer_name = None):
        if keys is None:
            keys = self.sp_names
        if layer_name is None:
            layer_name = 'layer1'
        keys = _ensure_list(keys)
        sp_point_indices = self.layers[layer_name].sp_point_indices[keys]
        all_full_indices = [indices for indices in sp_point_indices if len(indices) > 0]

        if len(all_full_indices) == 0:
            return np.array([], dtype=int)

        point_indices = np.concatenate(all_full_indices)
        return point_indices

    def get_points(self, keys, layer_name): #Numpy concatenate changes the dtype when a list is empty, so custom function.
        point_indices = self.get_point_indices(keys, layer_name)
        points = self.points[point_indices]
        return points
    
    def get_features(self, layer_name, feature_key, keys): #Numpy concatenate changes the dtype when a list is empty, so custom function.
        features = np.array(self.layers[layer_name].sp_features[feature_key])[keys]
        return features
    
    def delete_sp_index(self, indices):
        if self.sp_centers is not None:
            self.sp_centers = sm._delete(self.sp_centers, indices)

        for layer_name, layer in self.layers.items():
            layer.delete_sp_index(indices)


class SuperpointLayer():

    def __init__(self, layer_name):
        self.layer_name       = layer_name
        self.sp_point_indices = None
        self.sp_features      = KeyDict()

    def __getattr__(self, name): #allows shorthand reference to specific features
        if name in self.sp_features:
            return self.sp_features[name]
        else:
            return object.__getattribute__(self, name)
        
    def delete_sp_index(self, indices):
        if self.sp_point_indices is not None:
            self.sp_point_indices = sm._delete(self.sp_point_indices, indices)

        if self.sp_features is not None:
            for keyA, itemA in self.sp_features.items():
                if isinstance(itemA, list) or isinstance(itemA, np.ndarray):
                    self.sp_features[keyA] = sm._delete(itemA, indices)
                elif isinstance(itemA, dict):
                    for keyB, itemB in itemA.items():
                        self.sp_features[keyA][keyB] = sm._delete(itemB, indices)

# --------------------------------------
#             DESIGN CLASSES
# --------------------------------------
class LayerModifier():
    """
    A wrapper for layer modifying functions. Links methods with kwargs and allows predefined functions
    to be called by keyword look up.
    """

    def __init__(self, method, **kwargs):
        self.kwargs      = kwargs
        self.method      = None

        if callable(method): # if method is a function
            self.method  = method
        else:
            raise Exception('The "method" parameter must be a callable function.')
        
    def __call__(self, constructor):
        self.method(constructor, **self.kwargs)


class SuperpointLayerDesign():

    def __init__(self, layer_name):
        self.layer_name     = layer_name
        self.modifiers      = []

    def add_modifier(self, method, **kwargs):
        self.modifiers.append(LayerModifier(method, **kwargs))


class SuperpointSpaceDesign():

    def __init__(self):
        self.layer_designs  = []

    def add_design_layer(self, layer_name):
        new_layer_design = SuperpointLayerDesign(layer_name)
        self.layer_designs.append(new_layer_design)
        return new_layer_design
    
    def quality_check(self):
        if len(self.layer_designs) < 1:
            raise Exception('SuperpointSpaceDesign() requires at least 1 layer definition, none were defined.')
        for layer_design in self.layer_designs:
            if not layer_design.modifiers:
                raise Exception(f'Layer {layer_design.layer_name} has no modifier defined. Use the SuperpointLayerDesign.add_modifier(<modifier>) method to add a modifier to the layer.')
            

# --------------------------------------
#        CONSTRUCTOR CLASS
# --------------------------------------
class SuperpointSpaceConstructor():
    """
    DESCRIPTION: This class takes a SuperpointSpaceDesign object and points. It provides an API to space modifiers.
    """
    def __init__(self, sp_design, points):
        self.sp_design         = sp_design
        self.points            = points
        self.check_first_layer = False
        self.check_centers     = False
        self.check_point_inds  = []
        self.sp_space          = None
        self.cur_layer         = None
        self.cur_mod           = None

        if not isinstance(self.points, np.ndarray):
            raise Exception('<points> must be a numpy array.')

    def build_space(self, verbose = False):
        self.sp_design.quality_check()
        self.sp_space = SuperpointSpace(self.points)

        for layer_design in self.sp_design.layer_designs:
            self.cur_layer = SuperpointLayer(layer_design.layer_name)

            for modifier in layer_design.modifiers:
                if verbose: print(f"Building layer '{layer_design.layer_name}', method '{modifier.method.__name__}()'")
                self.cur_mod = f"Layer '{layer_design.layer_name}', method '{modifier.method.__name__}()'," # for error printing only
                modifier(self)
                
            self.sp_space.layers[layer_design.layer_name] = self.cur_layer
            self.check_first_layer = True

        #Quality control
        if not self.check_centers:
            raise Exception('The provided SuperpointSpaceDesign failed to set the superpoint centers. Please add a modifier to the first layer.')
        if len(self.check_point_inds) != len(self.sp_space.layers):
            raise Exception('The provided SuperpointSpaceDesign failed to set set superpoint point indices for all layers.')

        return self.sp_space

    def get_points(self):
        return self.sp_space.points
    
    def set_points(self):
        pass

    def get_sp_centers(self):
        if self.sp_space.sp_centers is None:
            raise Exception(f'{self.cur_mod} is requesting superpoint centers. The center of the superpoints have not yet been defined.')
        return self.sp_space.sp_centers
    
    def set_sp_centers(self, sp_centers, add_on = False):
        if not self.check_first_layer:
            if add_on and self.sp_space.sp_centers is not None:
                self.sp_space.sp_centers = np.vstack((self.get_sp_centers(), sp_centers))
            else:
                self.sp_space.sp_centers = sp_centers
            self.check_centers = True
        else:
            raise Exception(f'{self.cur_mod} is attempting to set superpoint centers. Centers can only be set by the first layer of the superpoint space.')

    def get_sp_point_indices(self, indices = None):
        if self.cur_layer.sp_point_indices is None:
            raise Exception(f'{self.cur_mod} is requesting the superpoint point indices. Superpoint point indices have not yet been defined.')
        if indices:
            return self.cur_layer.sp_point_indices[indices]
        else:
            return self.cur_layer.sp_point_indices
    
    def set_sp_point_indices(self, sp_point_indices, indices = None, add_on = False):
        if isinstance(sp_point_indices, list): #convert to array of lists
            new_sp_point_indices = np.empty(len(sp_point_indices), dtype=object)
            for i, index in enumerate(sp_point_indices):
                new_sp_point_indices[i] = index
            sp_point_indices = new_sp_point_indices

        if indices:
            self.cur_layer.sp_point_indices[indices] = sp_point_indices
            return
        
        if add_on and self.cur_layer.sp_point_indices is not None:
            self.cur_layer.sp_point_indices = np.hstack((self.get_sp_point_indices(), sp_point_indices))
        else:
            self.cur_layer.sp_point_indices = sp_point_indices
            
        if self.cur_layer.layer_name not in self.check_point_inds:
            self.check_point_inds.append(self.cur_layer.layer_name)

    def get_sp_feature(self, feature_key):
        if feature_key in self.cur_layer.sp_features:
            return self.cur_layer.sp_features[feature_key]
        else:
            return None
        
    def set_sp_feature(self, feature_key, feature):
        if len(feature) == len(self.sp_space): # a feature for each superpoint and no more.
            self.cur_layer.sp_features[feature_key] = feature
        else:
            raise Exception(f'{self.cur_mod} is attempting to add a feature of length {len(feature)} to a superpoint space of length {len(self.sp_space)}.')

    def delete_sp_index(self, indices):
        self.sp_space.delete_sp_index(indices)
        self.cur_layer.delete_sp_index(indices) # because layer is not added to superpoint until after construction TODO: can I add it to the sapce before construction?