import numpy as np
import random
import cv2
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy

from softgym.utils.pyflex_utils import center_object
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math
from softgym.utils.pyflex_utils import random_pick_and_place

from time import sleep

class FabricEnv(ClothEnv):
    def __init__(self, cached_states_path='mono_square_fabric.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._reward_mode = kwargs['reward_mode']
        

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def _reset(self):
        """ Right now only use one initial state"""
        self._set_to_flatten()
        # self._canonical_mask = self.get_cloth_mask()
        # self._target_img = self.render(mode='rgb')[:, :, :3]
        self.set_scene(self.cached_configs[self.current_config_id], self.cached_init_states[self.current_config_id])
        
        self._flatten_particel_positions = self.get_flattened_positions()
        self._flatten_coverage =  self.get_coverage(self._flatten_particel_positions)
        
        
        self._initial_particel_positions = self.get_particle_positions()
        self._initial_coverage = self.get_coverage(self._initial_particel_positions)
    

        # if self.action_mode == 'pickerpickplace':
        #     self.action_step = 0
        #     self._current_action_coverage = self._prior_action_coverage = self._initial_coverage

        if hasattr(self, 'action_tool'):
            self.action_tool.reset(np.asarray([0.2, 0.2, 0.2]))


        # if hasattr(self, 'action_tool'):
        #     curr_pos = pyflex.get_positions()
        #     #cx, cy = self._get_center_point(curr_pos)
        #     self.action_tool.reset(np.asarray([[0.2, 0.2, 0.2]]))

        # if hasattr(self, 'action_tool'):
        #     particle_pos = pyflex.get_positions().reshape(-1, 4)
        #     p1, p2, p3, p4 = self._get_key_point_idx()
        #     key_point_pos = particle_pos[(p1, p2), :3]
        #     middle_point = np.mean(key_point_pos, axis=0)
        #     self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            
        pyflex.step()
        self.init_covered_area = None
        return self._get_obs()