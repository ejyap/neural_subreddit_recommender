import numpy as np
import json
import h5py

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):

    bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] <= max_x)
    bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] <= max_y)
    bound_z = np.logical_and(points[:, 2] >= min_z, points[:, 2] <= max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

def load_h5_dataset(path, dataset):
    h5 = h5py.File(path)
    return h5.get(dataset)[()]

def load_json(path):
    with open(path) as f:
        return json.load(f)

