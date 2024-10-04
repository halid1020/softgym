import numpy as np
from softgym.utils.misc import vectorized_range, vectorized_meshgrid

def get_coverage(positions, particle_radius):
    """
    Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
    :param pos: Current positions of the particle states
    """

    min_x = np.min(positions[:, 0])
    min_y = np.min(positions[:, 2])
    max_x = np.max(positions[:, 0])
    max_y = np.max(positions[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = positions[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + particle_radius) / span[1]).astype(int), 100)

    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]