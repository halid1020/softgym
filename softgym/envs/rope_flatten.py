import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object

class RopeFlattenEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self._reward_mode = kwargs['reward_mode']
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.reset()
        self._corner_ids = [0, pyflex.get_positions().reshape(-1, 4).shape[0] - 1]
        

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()  

        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()
            


            pyflex.step()
            
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  

            print('config {}: {}'.format(i, config['camera_params']))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])
        pyflex.step()

        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min

        return self._get_obs(), None

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            if self.action_mode == 'pickerpickplace':
                self._wait_to_stabalise(render=True)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _wait_to_stabalise(self, max_wait_step=20, stable_vel_threshold=0.05, target_point=None, target_pos=None, render=False):
        for j in range(0, max_wait_step):
            curr_vel = pyflex.get_velocities()
            if target_point != None:
                curr_pos = pyflex.get_positions()
                curr_pos[target_point * 4: target_point * 4 + 3] = target_pos
                curr_vel[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)
            pyflex.step()
            if render:
                pyflex.render()
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                break

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        if self._reward_mode == "default_reward":
            curr_endpoint_dist = self._get_endpoint_distance()
            curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
            return curr_distance_diff
        
        elif self._reward_mode == "normalised_performance":
            return self.get_normalised_performance()
        else:
            raise NotImplementedError

    def get_particles_positions(self):
        return pyflex.get_positions().reshape(-1, 4)[:, :3]

    def get_normalised_performance(self):
        curr_endpoint_dist = self._get_endpoint_distance()

        return curr_endpoint_dist/self.rope_length

    def _get_info(self):
        
        curr_endpoint_dist = self._get_endpoint_distance()
        normalized_performance = self.get_normalised_performance()

        return {
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist
        }

    def get_corners_positions(self):
        all_particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        #print('num particles', len(all_particle_positions))
        return all_particle_positions[self._corner_ids]
    
    # TODO: refactor this function
    def _world_to_pixel(self, positions):
        camera_hight = 1.5 # TODO: magic number
        depths = camera_hight - positions[:, 1] #x, z, y
        pixel_to_world_ratio = 0.415 # TODO: magic number

        N = positions.shape[0]
        projected_pixel_positions_x = positions[:, 0]/pixel_to_world_ratio/depths #-1, 1
        projected_pixel_positions_y = positions[:, 2]/pixel_to_world_ratio/depths #-1, 1
        return  np.concatenate(
            [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
            axis=1)

    
    def get_corners_pixel_positions(self):
        return self._world_to_pixel(self.get_corners_positions())
 

    def get_particles_pixel_positions(self):
        return self._world_to_pixel(pyflex.get_positions().reshape(-1, 4)[:, :3])