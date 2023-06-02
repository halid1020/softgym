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

class ClothFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='mono_square_fabric.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._reward_mode = kwargs['reward_mode']
          
        
        if self.save_step_info:
            self.step_info = {}
        

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    
    def get_goal_observation(self):
        return self._target_img

    

    def _reset(self):
        """ Right now only use one initial state"""
        self._set_to_flatten()
        self.set_scene(self.cached_configs[self.current_config_id], self.cached_init_states[self.current_config_id])
        
        self._flatten_particel_positions = self.get_flatten_positions()
        self._flatten_coverage =  self.get_coverage(self._flatten_particel_positions)
        
        self._initial_particel_positions = self.get_particle_positions()
        self._initial_coverage = self.get_coverage(self._initial_particel_positions)
    

        if self.action_mode == 'pickerpickplace':
            self.action_step = 0
            self._current_action_coverage = self._prior_action_coverage = self._initial_coverage


        # if hasattr(self, 'action_tool'):
        #     curr_pos = pyflex.get_positions()
        #     #cx, cy = self._get_center_point(curr_pos)
        #     self.action_tool.reset(np.asarray([[0.2, 0.2, 0.2]]))

        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3]
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            
        pyflex.step()
        self.init_covered_area = None
        return self._get_obs(), None

    def _step(self, action):

        if self.save_step_info:
            self.step_info = {}

        self.control_step +=  self.action_tool.step(action)
        
        if self.save_step_info:
            self.step_info = self.action_tool.get_step_info()
            
            self.step_info['coverage'] = []
            self.step_info['reward'] = []
            steps = len(self.step_info['control_signal'])

            for i in range(steps):
                particle_positions = self.step_info['particle_pos'][i][:, :3]
                
                self.step_info['rgbd'][i] = cv2.resize(self.step_info['rgbd'][i], self.save_image_dim)
                self.step_info['reward'].append(self.compute_reward(particle_positions))
                self.step_info['coverage'].\
                    append(self.get_coverage(particle_positions))
                
                eval_data = self.evaluate(particle_positions)
                for k, v in eval_data.items():
                    if k not in self.step_info.keys():
                        self.step_info[k] = [v]
                    else:
                        self.step_info[k].append(v)


        if self.action_mode == 'pickerpickplace':
            self.action_step += 1
            self._wait_to_stabalise()

        else:
            self.tick_control_step()

        if self.save_step_info:
            self.step_info = {k: np.stack(v) for k, v in self.step_info.items()}

         ### Update parameters for quasi-static pick and place.
        if self.action_mode == 'pickerpickplace':
            self._prior_action_coverage = self._current_action_coverage
            self._current_action_coverage = self.get_coverage(self.get_particle_positions())


    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        if self._reward_mode == "distance_reward":
            return self._distance_reward(self.get_particle_positions())
        if self._reward_mode == "pixel_rmse":
            return self._pixel_reward(self.render())
        if self._reward_mode == "depth_ratio":
            return self._depth_reward(self.get_particle_positions())
        if self._reward_mode == "corner_and_depth_ratio":
            return self._corner_and_depth_reward(self.get_particle_positions())
        if self._reward_mode == "hoque_ddpg":
            return self._hoque_ddpg_reward()
        if self._reward_mode == 'normalised_coverage':
            return self._normalised_coverage()
        raise NotImplementedError

    def evaluate(self, particles=None):
        if particles is None:
            particles = self.get_particle_positions()

        target_coverage = self._flatten_coverage
        initial_coverage = self._initial_coverage
        current_coverage = self.get_coverage(particles)

        return {
            'normalised_improvement': (current_coverage - initial_coverage)/(target_coverage - initial_coverage),
            'normalised_coverage': (current_coverage/target_coverage),
            'wrinkle_pixel_ratio': self._get_wrinkle_pixel_ratio(particles)
        }
    
    
        

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    
  

    def _get_particle_positions(self):
        return pyflex.get_positions()

    def _set_particle_positions(self, pos):
        pyflex.set_positions(pos.flatten())
        pyflex.set_velocities(np.zeros_like(pos))
        pyflex.step()
        self._current_coverage = self.get_covered_area(self.get_particle_positions())
        
    
    def _distance_reward(self, particle_pos):
        min_distance = self.get_performance_value(particle_pos)
        return math.exp(-min_distance/10)

    def _pixel_reward(self, img):
        return ((1 - math.sqrt(np.mean((img/255.0-self._target_img/255.0)**2))) -0.5) * 2

    def _depth_reward(self, particle_pos):

        return len(np.where(particle_pos[:, 1] <= 0.008)[0])/len(particle_pos)

    def _corner_and_depth_reward(self, particle_pos):
        target_corner_positions =  self.get_corner_positions()
        visibility = self.get_visibility(target_corner_positions)
        count = np.count_nonzero(visibility)
        reward = count * 0.1
        depth_reward = self._depth_reward(particle_pos)
        
        if depth_reward >= 0.5:
            depth_reward = (depth_reward - 0.5) * 2
            reward += 0.6 * depth_reward

        return reward

    

    def _hoque_ddpg_reward(self):

        reward = (self._current_action_coverage - self._prior_action_coverage)/self._flatten_coverage
        
        bonus = 0
        if abs(self._current_action_coverage - self._prior_action_coverage) <= 1e-4:
            reward = -0.05
        
        if self._current_action_coverage /self._flatten_coverage > 0.92:
            bonus = 1

        reward += bonus
        

        return reward

    