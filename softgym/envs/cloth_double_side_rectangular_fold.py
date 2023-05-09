# from turtle import shape
# from matplotlib.pyplot import step
import numpy as np
# import pyflex
# from copy import deepcopy
# from softgym.envs.cloth_env import ClothEnv
# from softgym.utils.pyflex_utils import center_object
# import cv2

from softgym.envs.cloth_fold import ClothFoldEnv

class ClothDoubleSideRectangularFoldEnv(ClothFoldEnv):

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
        for _ in range(2):
            X = particle_grid_idx.shape[0]
            x_split, x_split_2 = X // 4, X // 2
            group_a = np.concatenate([particle_grid_idx[:x_split].flatten(), particle_grid_idx[X-x_split:].flatten()])
            group_b = np.concatenate([
                np.flip(particle_grid_idx[x_split:2*x_split], axis=0).flatten(),
                np.flip(particle_grid_idx[X-x_split:X], axis=0).flatten()])
            
            group_a = np.concatenate([group_a, particle_grid_idx[:x_split_2].flatten()])
            group_b = np.concatenate([group_b, np.flip(particle_grid_idx[X-x_split_2:], axis=0).flatten()])
            
                
            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)

        return res_a, res_b 