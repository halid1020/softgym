from turtle import shape
from matplotlib.pyplot import step
import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
import cv2

class ClothRectangularFoldEnv(ClothFoldEnv):

    def __init__(self, cached_states_path='cloth_folding_tmp.pkl', **kwargs):
        #self.cloth_particle_radius = kwargs['particle_radius']
        
        super().__init__(**kwargs)


    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        
        res_a, res_b = super()._reset()

        ### Side folding utilities
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][0], config['ClothSize'][1]).T  # Reversed index here
        vertical_flip_particle_grid_idx = np.flip(particle_grid_idx, 1)

        cloth_dimx, cloth_dimy = config['ClothSize']
        x_split, ysplit= cloth_dimx // 2, cloth_dimy // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        self.fold_group_a_flip = vertical_flip_particle_grid_idx[:ysplit, :].flatten()
        self.fold_group_b_flip = np.flip(vertical_flip_particle_grid_idx, axis=1)[:ysplit, :].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1

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
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            edge_ids = self.get_edge_ids()
            edge_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in edge_ids]
            edge_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in edge_ids]

            return min(np.mean(edge_distance_1), np.mean(edge_distance_2))
        else:
            raise NotImplementedError

    def _largest_edge_distance(self, particles=None):
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            edge_ids = self.get_edge_ids()
            edge_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in edge_ids]
            edge_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in edge_ids]

            return min(np.max(edge_distance_1), np.max(edge_distance_2))
        else:
            raise NotImplementedError

    def _corner_distance(self, particles=None):

        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            corner_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in self._corner_ids]
            corner_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in self._corner_ids]

            return min(np.max(corner_distance_1), np.max(corner_distance_2))
        else:
            raise NotImplementedError

    def _mean_particle_distance(self, particles=None):

        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            return min(np.mean(distances_1), np.mean(distances_2))
        else:
            raise NotImplementedError



    def _largest_particle_distance(self, particles=None):
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            return min(np.max(distances_1), np.max(distances_2))
        else:
            raise NotImplementedError
