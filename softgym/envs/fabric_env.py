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
        #self._reward_mode = kwargs['reward_mode']
        

        if self.use_cached_states:
            self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def get_edge_ids(self):
        config = self.get_current_config()
        cloth_dimx,  cloth_dimy = config['ClothSize']
        edge_ids = [i for i in range(cloth_dimx)]
        edge_ids.extend([i*cloth_dimx for i in range(1, cloth_dimy)])
        edge_ids.extend([(i+1)*cloth_dimx-1 for i in range(1, cloth_dimy)])
        edge_ids.extend([(cloth_dimy-1)*cloth_dimx + i for i in range(1, cloth_dimx-1)])
        return edge_ids

    def _generate_env_config(self, index):


        config = deepcopy(self.get_default_config().copy())
        random_state = np.random.RandomState(index)
        if 'size' in self.context:
            width = random_state.uniform(
                self.context['size']['width']['lower_bound'], 
                self.context['size']['width']['upper_bound'])
            if 'rectangular' in self.context and self.context['rectangular']:
                length = random_state.uniform(
                    self.context['size']['length']['lower_bound'],
                    self.context['size']['length']['upper_bound'])
            else:
                length = width
            
            # print('cloth width (m)', width)
            # print('cloth length (m)', length)

            config['ClothSize'] = [
                int(width / self.cloth_particle_radius), 
                int(length / self.cloth_particle_radius)]
        
        if 'colour' in self.context:
            config['front_colour'] = random_state.uniform(
                np.array(self.context['colour']['front_colour']['lower_bound']), 
                np.array(self.context['colour']['front_colour']['upper_bound']))

            if 'colour_mode' in self.context and self.context['colour_mode'] == 'both_same':
                config['back_colour'] = config['front_colour']
                #print('here here')
            else:
                config['back_colour'] = random_state.uniform(
                    np.array(self.context['colour']['back_colour']['lower_bound']),
                    np.array(self.context['colour']['back_colour']['upper_bound']))
        
        if 'flip_face' in self.context and random_state.random() < self.context['flip_face']:
            config['front_colour'], config['back_colour'] = config['back_colour'], config['front_colour']
        

        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
        
        cloth_dimx, cloth_dimy = config['ClothSize']
        #print('config to Set !!!', config)
        self.set_scene(config)
        #self.action_tool.reset([0., -1., 0.])
        
        self._set_to_flatten()

        if index != 0:
            
            # initilise a sampler regarding the idex
            random_state = np.random.RandomState(index)

            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']:  # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            pyflex.step()

            num_particle = cloth_dimx * cloth_dimy

            # Pick up the cloth and wait to stablize
            print('state', self.context['state'])
            if self.context['state']:
                pickpoint = self.random_state.randint(0, num_particle - 1)
                curr_pos = pyflex.get_positions()
                original_inv_mass = curr_pos[pickpoint * 4 + 3]
                curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                pickpoint_pos[1] += self.random_state.random()*0.4
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                pyflex.set_positions(curr_pos.flatten())
                self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)
                

                # Drop the cloth and wait to stablize
                curr_pos = pyflex.get_positions()
                curr_pos[pickpoint * 4 + 3] = original_inv_mass
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                pyflex.set_positions(curr_pos.flatten())          
                self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, None, None)

                center_object(random_state, self.context['position'])

                # Drag the cloth and wait to stablise
                if random_state.random() < 0.7:
                    pickpoint = random_state.randint(0, num_particle - 1)
                    curr_pos = pyflex.get_positions()
                    original_inv_mass = curr_pos[pickpoint * 4 + 3]
                    curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                    pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                    pickpoint_pos[0] += (np.random.random(1)*2 - 1)*0.3
                    pickpoint_pos[2] += (np.random.random(1)*2 - 1)*0.3
                    pickpoint_pos[1] = 0.1
                    curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                    pyflex.set_positions(curr_pos.flatten())
                    self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)


                    # Drop the cloth and wait to stablize
                    curr_pos = pyflex.get_positions()
                    curr_pos[pickpoint * 4 + 3] = original_inv_mass
                    pyflex.set_positions(curr_pos.flatten())          
                    self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, None, None)

                    center_object(random_state, self.context['position'])


            while True:
                center_object(random_state, self.context['position'])
                if self.context['rotation']:
                    angle = random_state.rand(1) * np.pi * 2
                    self._rotate_particles(angle)
                self._wait_to_stabalise()
                
                ## check the boundary of the cloth particle see if it is beyond -1 and 1 if so, redo the centering and rotation
                positions = self.get_particle_positions()
                N = positions.shape[0]
                depths = self.camera_height - positions[:, 1] #x, z, y
                pixel_to_world_ratio = self.pixel_to_world_ratio # TODO: magic number
                projected_pixel_positions_x = positions[:, 0]/pixel_to_world_ratio/depths #-1, 1
                projected_pixel_positions_y = positions[:, 2]/pixel_to_world_ratio/depths #-1, 1
                projected_pixel_positions = np.concatenate(
                    [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
                    axis=1)
                # print('max projected_pixel_positions', np.max(projected_pixel_positions))
                # print('min projected_pixel_positions', np.min(projected_pixel_positions))

                if 'all_visible' in self.context and self.context['all_visible']:
                    if np.alltrue(np.abs(projected_pixel_positions) < 1):
                        break
                else:
                    break

            


            
            

            # if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
            #     self.action_tool.reset(np.asarray([0.5, 0.2, 0.5]))

        pyflex.step()
        pyflex.render()
        
        return config, self.get_state()

    def get_corner_positions(self, position=None):
        if position == None:
            positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        return positions[self._corner_ids]
    
    def get_keypoint_positions(self, position=None):
        return self.get_corner_positions(position)
    
    def _set_to_flatten(self):
        new_pos = self._flatten_pos()
        pyflex.set_positions(new_pos.flatten())
        flag = self.save_control_step_info
        self.save_control_step_info = False
        self.wait_until_stable()
        self.save_control_step_info = flag
        self._canonical_mask = self.get_cloth_mask(camera_name=self.current_camera)
        self.flatten_obs = self._get_obs()
        self._flatten_coverage =  self.get_coverage()
        #logging.info('[softgym, cloth_env] flatten_coverage: {}'.format(self._flatten_coverage))
        self._flatten_corner_positions = self.get_corner_positions()
        self._flatten_edge_positions = self.get_edge_positions()
        new_pos = self.get_particle_positions()

        return self.get_coverage(new_pos)
    
    def _reset(self):
        """ Right now only use one initial state"""
        # self._set_to_flatten()
        # self._canonical_mask = self.get_cloth_mask()
        # self._target_img = self.render(mode='rgb')[:, :, :3]
        
        if self.use_cached_states:
            assert self.current_config_id < len(self.cached_configs), \
                'current_config_id {} should be less than size of cached_configs {}'.format(
                    self.current_config_id, len(self.cached_configs))
            
            self.set_scene(
                self.cached_configs[self.current_config_id], 
                self.cached_init_states[self.current_config_id])
            self._goal = self.set_to_flatten()
            self.set_scene(
                self.cached_configs[self.current_config_id], 
                self.cached_init_states[self.current_config_id])
            self.current_config = self.cached_configs[self.current_config_id]
        else:
            config, state = self._generate_env_config(self.current_config_id)
            self.set_scene(config, state)
            self.current_config = config
       
        
        
        self._initial_particel_positions = self.get_particle_positions()
        self._initial_coverage = self.get_coverage(self._initial_particel_positions)
    
        if hasattr(self, 'action_tool'):
            self.action_tool.reset(np.asarray(self.picker_initial_pos))

            
        pyflex.step()
        self.init_covered_area = None
        return self._get_obs()
    
    def get_cloth_dim(self):
        H, W = self.get_cloth_size()
        return H * self.cloth_particle_radius, W * self.cloth_particle_radius