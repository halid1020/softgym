# from turtle import shape
# from matplotlib.pyplot import step
import numpy as np
# import pyflex
# from copy import deepcopy
# from softgym.envs.cloth_env import ClothEnv
# from softgym.utils.pyflex_utils import center_object
# import cv2

from softgym.envs.cloth_fold import ClothFoldEnv

class ClothSideFoldEnv(ClothFoldEnv):

    def __init__(self, cached_states_path='cloth_folding_tmp.pkl', **kwargs):
        #self.cloth_particle_radius = kwargs['particle_radius']
        
        super().__init__(cached_states_path, **kwargs)


    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        
        res_a, res_b = super()._reset()

        ### side folding utilities
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][0], config['ClothSize'][1]).T  # Reversed index here

        self.fold_groups = []
        for _ in range(4):
            X = particle_grid_idx.shape[0]
            x_split = X // 4
            group_a = particle_grid_idx[:, :x_split].flatten()
            group_b = np.flip(particle_grid_idx[:, x_split:2*x_split], axis=1).flatten()
            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)

        return res_a, res_b 
    
    def _get_distance(self, positions, group_a, group_b):
        if positions is None:
            position = self.get_particle_positions()
        cols = [0, 1, 2]
        pos_group_a = position[np.ix_(group_a, cols)]
        pos_group_b = position[np.ix_(group_b, cols)]
        distance = np.linalg.norm(pos_group_a-pos_group_b, axis=1)
        return distance
    
    def _mean_edge_distance(self, particles=None):
        
        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            distances.append(self._get_distance(particles, group_a, group_b))
            edge_distance.append(np.mean([distances[-1][i] for i, p in enumerate(group_a) if p in edge_ids]))
        
        return np.min(edge_distance)

    def _largest_edge_distance(self, particles=None):

        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            distances.append(self._get_distance(particles, group_a, group_b))
            edge_distance.append(np.max([distances[-1][i] for i, p in enumerate(group_a) if p in edge_ids]))
        
        return np.min(edge_distance)


    def _largest_corner_distance(self, particles=None):
        distances = []
        corner_distance = []
        for group_a, group_b in self.fold_groups:
            distances.append(self._get_distance(particles, group_a, group_b))
            corner_distance.append(np.max([distances[-1][i] for i, p in enumerate(group_a) if p in self._corner_ids]))

        return np.min(corner_distance)


    def _mean_particle_distance(self, particles=None):

        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.mean(self._get_distance(particles, group_a, group_b)))
        
        return np.min(distances)


    def _largest_particle_distance(self, particles=None):
        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.max(self._get_distance(particles, group_a, group_b)))
        
        return np.min(distances)
