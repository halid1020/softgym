import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

DIAGONAL_FOLDING_SUCCESS_THRESHOLD = 0.045
DIAGNOL_CROSS_FOLDING_SUCCESS_THRESHOLD = 0.08
CROSS_FOLDING_SUCCESS_THRESHOLD = 0.07
DOUBLE_SIDE_FOLDING_SUCCESS_THRESHOLD = 0.045
SIDE_FOLDING_SUCCESS_THRESHOLD = 0.045
RECTANGLUAR_FOLDING_SUCCESS_THRESHOLD = 0.045
ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD = 0.045
ONE_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD = 0.045
DOUBLE_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD = 0.045
CORNERS_EDGE_INWARD_FOLDING_SUCCESS_THRESHOLD = 0.045
FOLDING_IoU_THRESHOLD = 0.7

def get_wrinkle_pixel_ratio(rgb, mask):
    
    rgb = cv2.resize(rgb, (128, 128))
    #mask =  cv2.resize(mask, (128, 128)) 
    

    if mask.dtype != np.uint8:  # Ensure mask has a valid data type (uint8)
        mask = mask.astype(np.uint8)


    # Use cv2 edge detection to get the wrinkle ratio.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # plt.imshow(edges)
    # plt.show()

    masked_edges = cv2.bitwise_and(edges, mask)
    # plt.imshow(masked_edges)
    # plt.show()

    wrinkle_ratio = np.sum(masked_edges) / np.sum(mask)

    return wrinkle_ratio

def get_canonical_IoU(mask, canonical_mask):
    intersection = np.sum(np.logical_and(mask, canonical_mask))
    union = np.sum(np.logical_or(mask, canonical_mask))
    return intersection/union

def get_canonical_hausdorff_distance(mask, canonical_mask):
    hausdorff_distance = directed_hausdorff(mask, canonical_mask)[0]

    return hausdorff_distance



def rotation_matrix_z(theta):
    theta = np.asscalar(theta) if isinstance(theta, np.ndarray) else theta
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def objective_function(cur_particles, goal_particles, theta):
    R = rotation_matrix_z(theta)
    rotated_particles = np.dot(cur_particles, R.T)  # Apply rotation
    distances = np.linalg.norm(rotated_particles - goal_particles, axis=1)
    return np.sum(distances**2)