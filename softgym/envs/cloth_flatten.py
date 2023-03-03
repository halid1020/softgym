import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
from softgym.utils.pyflex_utils import random_pick_and_place

from time import sleep

class ClothFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._reward_mode = kwargs['reward_mode']
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.reset()
        self._set_to_flatten()
        self._initial_covered_area = None  # Should not be used until initialized
        

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

    def generate_env_variation(self, num_variations=1, vary_cloth_size=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            self._set_to_flatten()
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
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            pickpoint_pos[1] += random.random()*0.4
            pyflex.set_positions(curr_pos)
            self._wait_to_stabalise(max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)
            

            # Drop the cloth and wait to stablize
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = original_inv_mass
            pyflex.set_positions(curr_pos)          
            self._wait_to_stabalise(max_wait_step, stable_vel_threshold, None, None)

            center_object()

            # Drag the cloth and wait to stablise
            if random.random() < 0.7:
                pickpoint = random.randint(0, num_particle - 1)
                curr_pos = pyflex.get_positions()
                original_inv_mass = curr_pos[pickpoint * 4 + 3]
                curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                pickpoint_pos[0] += (np.random.random(1)*2 - 1)*0.3
                pickpoint_pos[2] += (np.random.random(1)*2 - 1)*0.3
                pickpoint_pos[1] = 0.1
                pyflex.set_positions(curr_pos)
                self._wait_to_stabalise(max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)


                # Drop the cloth and wait to stablize
                curr_pos = pyflex.get_positions()
                curr_pos[pickpoint * 4 + 3] = original_inv_mass
                pyflex.set_positions(curr_pos)          
                self._wait_to_stabalise(max_wait_step, stable_vel_threshold, None, None)

                center_object()


            
            

            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            
            pyflex.step()
            
                
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = self._set_to_flatten()  # Record the maximum flatten area

            print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states
    
    def get_goal_observation(self):
        return self._target_img

    def get_target_corner_positions(self):
        # 4*3
        return self._target_corner_positions
    
    def _flatten_pos(self):
        cloth_dimx, cloth_dimz = self.get_current_config()['ClothSize']
        
        N = cloth_dimx * cloth_dimz
        px = np.linspace(0, cloth_dimx * self.cloth_particle_radius, cloth_dimx)
        px -= cloth_dimx * self.cloth_particle_radius/2 
        py = np.linspace(0, cloth_dimz * self.cloth_particle_radius, cloth_dimz)
        py -= cloth_dimz * self.cloth_particle_radius/2

        xx, yy = np.meshgrid(px, py)
        L = len(xx)
        W = len(xx[0])
        self._corner_ids = [0, W-1, (L-1)*W, L*W-1]
        #print('_corner_ids', self._corner_ids)
        #print('corner ids', self._corner_ids)
        new_pos = np.empty(shape=(N, 4), dtype=float)
        new_pos[:, 0] = xx.flatten()
        new_pos[:, 1] = self.cloth_particle_radius
        new_pos[:, 2] = yy.flatten()
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        self._target_pos = new_pos.copy()

        return new_pos.copy()
    
    def flatten_pos(self):
        pos = self._flatten_pos()
        return pos.reshape(-1, 4)[:, :3].copy()


    def _set_to_flatten(self):
        # self._get_current_covered_area(pyflex.get_positions().reshape(-))
        new_pos = self._flatten_pos()
        
        pyflex.set_positions(new_pos.flatten())
        #pyflex.step()
        self._target_img = self._get_obs()['image']
        self._target_corner_positions = self._get_corner_positions()

        new_pos = self.get_particle_positions()

        return self.get_covered_area(new_pos)
    
    def _get_corner_positions(self):
        all_particle_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        #print('particles num', len(all_particle_positions))
        # print('first particle position', all_particle_positions[0])
        # print('last particle position', all_particle_positions[-1])
        #print('num particles', len(all_particle_positions))
        return all_particle_positions[self._corner_ids]
    
    def _get_info(self):
        pass

    def get_corner_positions(self, pos):
        return pos[self._corner_ids]
    

    def _reset(self):
        """ Right now only use one initial state"""
        self._initial_particel_pos = self.get_particle_positions()
        self._initial_covered_area = self.get_covered_area(self._initial_particel_pos)
        self._current_coverage_area = self._prior_coverage_area = self._initial_covered_area
        self._target_particel_pos = self.flatten_pos()
        self._target_covered_area =  self.get_covered_area(self._target_particel_pos)


        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        pyflex.step()
        self.init_covered_area = None
        return self._get_obs(), None

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode == 'pickerpickplace':
            self._wait_to_stabalise(render=True)
       
        if self.action_mode in ['sawyer', 'franka']:
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

        self._prior_coverage_area = self._current_coverage_area
        self._current_coverage_area = self.get_covered_area(self.get_particle_positions())
        return

    def get_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        #pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)

        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]


    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    
    def get_particle_positions(self):
        pos = pyflex.get_positions()
        pos = pos.reshape(-1, 4)[:, :3].copy()
        return pos

    def _get_particle_positions(self):
        return pyflex.get_positions()

    def _set_particle_positions(self, pos):
        pyflex.set_positions(pos.flatten())
        pyflex.set_velocities(np.zeros_like(pos))
        pyflex.step()
        self._current_coverage_area = self.get_covered_area(self.get_particle_positions())
        
    
    def _distance_reward(self, particle_pos):
        min_distance = self.get_performance_value(particle_pos)
        return math.exp(-min_distance/10)

    def _pixel_reward(self, img):
        return ((1 - math.sqrt(np.mean((img/255.0-self._target_img/255.0)**2))) -0.5) * 2

    def _depth_reward(self, particle_pos):

        return len(np.where(particle_pos[:, 1] <= 0.008)[0])/len(particle_pos)

    def _corner_and_depth_reward(self, particle_pos):
        target_corner_positions =  self._get_corner_positions()
        visibility = self.get_visibility(target_corner_positions)
        count = np.count_nonzero(visibility)
        reward = count * 0.1
        depth_reward = self._depth_reward(particle_pos)
        
        if depth_reward >= 0.5:
            depth_reward = (depth_reward - 0.5) * 2
            reward += 0.6 * depth_reward

        return reward

    def get_cloth_mask(self, pixel_size=(64, 64)):
        depth_images = self.render(mode='rgbd')[:, :, 3]
        # depth_images = np.expand_dims(depth_images, 2)
        # print('depth image shape', depth_images.shape)
        if pixel_size != (720,720):
            depth_images = cv2.resize(depth_images, pixel_size, interpolation=cv2.INTER_LINEAR)
        # print('depth image after shape', depth_images.shape)
       
        mask = (1.35 < depth_images) & (depth_images < 1.499)
        return mask



    def get_visibility(self, positions):
        # TODO: need to refactor this, so bad.
        # This has to be online.

        N = positions.shape[0]
        
        visibility = [False for _ in range(N)]

        camera_hight = 1.5 # TODO: magic number
        depths = camera_hight - positions[:, 1] #x, z, y
        pixel_to_world_ratio = 0.415 # TODO: magic number
        projected_pixel_positions_x = positions[:, 0]/pixel_to_world_ratio/depths #-1, 1
        projected_pixel_positions_y = positions[:, 2]/pixel_to_world_ratio/depths #-1, 1
        projected_pixel_positions = np.concatenate(
            [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
            axis=1)

        depth_images = self.render(mode='rgbd')[:, :, 3]
        depth_images = cv2.resize(depth_images, (128, 128))
        #print(depth_images.shape)

        for i in range(N):
            x, y = projected_pixel_positions[i][0],  projected_pixel_positions[i][1]
            if x < -1 or x > 1 or y < -1 or y > 1:
                continue
            x_ = int((y + 1)/2 * 128)
            y_ = int((x + 1)/2 * 128)
            if depths[i] < depth_images[x_][y_] + 1e-6:
                visibility[i] = True
            
        

        return np.asarray(visibility)

    def _hoque_ddpg_reward(self):

        reward = (self._current_coverage_area - self._prior_coverage_area)/self._target_covered_area
        
        bonus = 0
        if abs(self._current_coverage_area - self._prior_coverage_area) <= 1e-4:
            reward = -0.05
        
        if self._current_coverage_area/self._target_covered_area > 0.92:
            bonus = 1 #reward += 5
        
        reward += bonus
        
         # TODO: -5 for out-of-bound

        return reward

    def _normalised_coverage(self):
        return self._current_coverage_area/self._target_covered_area


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

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p


    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker)  * -1 # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps