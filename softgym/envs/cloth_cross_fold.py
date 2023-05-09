# from turtle import shape
# from matplotlib.pyplot import step
import numpy as np
# import pyflex
# from copy import deepcopy
# from softgym.envs.cloth_env import ClothEnv
# from softgym.utils.pyflex_utils import center_object
# import cv2

from softgym.envs.cloth_fold import ClothFoldEnv

class ClothCrossFoldEnv(ClothFoldEnv):

    def __init__(self, cached_states_path='cloth_folding_tmp.pkl', **kwargs):
        #self.cloth_particle_radius = kwargs['particle_radius']
        
        super().__init__(cached_states_path, **kwargs)


    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        
        res_a, res_b = super()._reset()

        ### side folding utilities
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        self.fold_groups = []
        X, Y = particle_grid_idx.shape
        x_split, y_split = X // 2, Y // 2
        group_a = np.concatenate([particle_grid_idx[:x_split, :y_split].flatten() for _ in range(3)])
        group_b = np.concatenate([
            np.flip(particle_grid_idx[x_split:2*x_split, :y_split], axis=0).flatten(), 
            np.flip(particle_grid_idx[:x_split, y_split:2*y_split], axis=1).flatten(),
            np.flip(np.flip(particle_grid_idx[x_split:2*x_split, y_split:2*y_split], axis=0), axis=1).flatten()])
    

        self.fold_groups.append((group_a, group_b))


        return res_a, res_b 