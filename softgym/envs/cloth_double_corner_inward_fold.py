# from turtle import shape
# from matplotlib.pyplot import step
import numpy as np
# import pyflex
# from copy import deepcopy
# from softgym.envs.cloth_env import ClothEnv
# from softgym.utils.pyflex_utils import center_object
# import cv2

from softgym.envs.cloth_fold import ClothFoldEnv

class ClothDoubleCornerInwardFoldEnv(ClothFoldEnv):

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
        X, Y = particle_grid_idx.shape
        ## Only allow square fabric
        assert X == Y, "Only allow square fabric"

        self.fold_groups = []
        for _ in range(2):
            x_split = X // 2
            upper_triangle_ids = np.triu_indices(x_split)

            group_a = np.concatenate([
                particle_grid_idx[:x_split, :x_split][upper_triangle_ids].flatten(), 
                particle_grid_idx[X-x_split:, X-x_split:][upper_triangle_ids].flatten()])
            group_b = np.concatenate([
                particle_grid_idx[:x_split, :x_split].T[upper_triangle_ids].flatten(),  
                particle_grid_idx[X-x_split:, X-x_split:].T[upper_triangle_ids].flatten()])

            self.fold_groups.append((group_a, group_b))
            particle_grid_idx = np.rot90(particle_grid_idx)

        return res_a, res_b

