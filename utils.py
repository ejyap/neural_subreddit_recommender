import numpy as np

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):

    bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] <= max_x)
    bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] <= max_y)
    bound_z = np.logical_and(points[:, 2] >= min_z, points[:, 2] <= max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter