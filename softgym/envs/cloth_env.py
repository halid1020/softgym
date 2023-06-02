import numpy as np
import cv2
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import  Picker, PickerPickPlace
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object

import matplotlib.pyplot as plt

class ClothEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', 
        picker_radius=0.02, picker_threshold=0.002, particle_radius=0.00625, 
        motion_trajectory='normal',
         **kwargs):
        
        self.cloth_particle_radius = particle_radius
        
        super().__init__(**kwargs)

        self.render_mode = render_mode
        self.action_mode = action_mode
        self.pixel_to_world_ratio = 0.4135

        # Context
        self.recolour_config = kwargs['recolour_config']
        if self.use_cached_states == False:
            self.context = kwargs['context']
        self.context_random_state = np.random.RandomState(kwargs['random_seed'])
        

        #assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb', 'cam_rgbd']
        assert action_mode in ['velocity_control', 'pickerpickplace', 'pickerpickplace1', 'sawyer', 'franka', 'picker_qpg']
        self.observation_mode = observation_mode

        if action_mode == 'velocity_control':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius, picker_threshold=picker_threshold,
                                      picker_low=kwargs['picker_low'], picker_high=kwargs['picker_high'], save_step_info=kwargs['save_step_info'])
            self.action_space = self.action_tool.action_space
            self.picker_radius = picker_radius
        
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(
                num_picker=num_picker, 
                particle_radius=particle_radius, 
                env=self, picker_threshold=picker_threshold, 
                picker_radius=picker_radius,
                motion_trajectory=motion_trajectory,
                camera_depth=self.get_current_config()['camera_params']['default_camera']['pos'][1],
                **kwargs)
            self.action_step = 0
            self.action_horizon = kwargs['action_horizon']

            self.action_space = self.action_tool.action_space
            assert self.action_repeat == 1


        if ('state' not in observation_mode.keys()) or (observation_mode['state'] == None):
            pass
        elif observation_mode['state'] == 'key_point':
            sts_dim = len(self._get_key_point_idx()) * 3
        elif observation_mode['state'] == 'positions':
            config = self.get_current_config()
            dimx, dimy = config['ClothSize']
            sts_dim = dimx * dimy * 3
            self.particle_obs_dim = sts_dim
        elif observation_mode['state'] == 'corner_pixel':
            sts_dim = 4*2
        else:
            raise NotImplementedError
        
        if ('state' in observation_mode.keys()) and (observation_mode['state'] != None):
            self.state_space = Box(np.array([-np.inf] * sts_dim), np.array([np.inf] * sts_dim), dtype=np.float32)

        if observation_mode['image'] == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)
                                    
        elif observation_mode['image'] == 'cam_rgbd':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 4),
                                         dtype=np.float32)

        elif observation_mode['image'] == 'cam_d':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 1),
                                         dtype=np.float32)
            
    def get_action_space(self):
        return self.action_space
            
    def generate_env_variation(self, num_variations=1): #, vary_cloth_size=False)
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.02  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config().copy()

       

        for i in range(num_variations):
            config = deepcopy(default_config)
            if 'size' in self.context:
                width = self.context_random_state.uniform(
                    self.context['size']['width']['lower_bound'], 
                    self.context['size']['length']['upper_bound'])
                length = self.context_random_state.uniform(
                    self.context['size']['length']['lower_bound'],
                    self.context['size']['length']['upper_bound'])
                
                config['ClothSize'] = [
                    int(width / self.cloth_particle_radius), 
                    int(length / self.cloth_particle_radius)]
            
            if 'colour' in self.context:
                config['front_colour'] = self.context_random_state.uniform(
                    np.array(self.context['colour']['front_colour']['lower_bound']), 
                    np.array(self.context['colour']['front_colour']['upper_bound']))
                config['back_colour'] = self.context_random_state.uniform(
                    np.array(self.context['colour']['back_colour']['lower_bound']),
                    np.array(self.context['colour']['back_colour']['upper_bound']))
            
                
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            # if vary_cloth_size:
            #     cloth_dimx, cloth_dimy = self._sample_cloth_size()
            #     config['ClothSize'] = [cloth_dimx, cloth_dimy]
            # else:
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
            if self.context['state']:
                pickpoint = self.context_random_state.randint(0, num_particle - 1)
                curr_pos = pyflex.get_positions()
                original_inv_mass = curr_pos[pickpoint * 4 + 3]
                curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                pickpoint_pos[1] += self.context_random_state.random()*0.4
                pyflex.set_positions(curr_pos.flatten())
                self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)
                

                # Drop the cloth and wait to stablize
                curr_pos = pyflex.get_positions()
                curr_pos[pickpoint * 4 + 3] = original_inv_mass
                pyflex.set_positions(curr_pos.flatten())          
                self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, None, None)

                center_object(self.context_random_state, self.context['position'])

                # Drag the cloth and wait to stablise
                if self.context_random_state.random() < 0.7:
                    pickpoint = self.context_random_state.randint(0, num_particle - 1)
                    curr_pos = pyflex.get_positions()
                    original_inv_mass = curr_pos[pickpoint * 4 + 3]
                    curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
                    pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
                    pickpoint_pos[0] += (np.random.random(1)*2 - 1)*0.3
                    pickpoint_pos[2] += (np.random.random(1)*2 - 1)*0.3
                    pickpoint_pos[1] = 0.1
                    pyflex.set_positions(curr_pos.flatten())
                    self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, pickpoint, pickpoint_pos)


                    # Drop the cloth and wait to stablize
                    curr_pos = pyflex.get_positions()
                    curr_pos[pickpoint * 4 + 3] = original_inv_mass
                    pyflex.set_positions(curr_pos.flatten())          
                    self._wait_to_stabalise() #max_wait_step, stable_vel_threshold, None, None)

                    center_object(self.context_random_state, self.context['position'])

            center_object(self.context_random_state, self.context['position'])
            if self.context['rotation']:
                angle = self.context_random_state.rand(1) * np.pi * 2
                self._rotate_particles(angle)

            
            

            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                self.action_tool.reset(np.asarray([0.5, 0.2, 0.5]))
            
            pyflex.step()
            
                
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = self._set_to_flatten()  # Record the maximum flatten area

            print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states
    
    def _rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4).copy()
        center = np.mean(pos[:, [0, 2]], axis=0, keepdims=True) 
        pos[:, [0, 2]] -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos[:, [0, 2]] += center
        pyflex.set_positions(new_pos.flatten())

    def get_particle_pos(self):
        pos = pyflex.get_positions()
        return pos.reshape(-1, 4).copy()

    def set_pos(self, particle_pos, picker_pos):
        pyflex.set_positions(particle_pos.flatten())
        pyflex.set_shape_states(picker_pos)
        pyflex.step()
        if self._render:
            pyflex.render()

    def get_particle_positions(self):
        return self.get_particle_pos()[:, :3].copy()

    def get_picker_pos(self):
        return self.action_tool.get_picker_pos()

    def get_corner_positions(self, particle_positions=None):
        if particle_positions is None:
            particle_positions = self.get_particle_positions()
        return particle_positions[self._corner_ids]

    def get_flatten_corner_positions(self):
        return self._flatten_corner_positions


    def get_cloth_mask(self, pixel_size=(64, 64)):
        depth_images = self.render(mode='rgbd')[:, :, 3]
        if pixel_size != (720,720):
            depth_images = cv2.resize(depth_images, pixel_size, interpolation=cv2.INTER_LINEAR)
        mask = (1.35 < depth_images) & (depth_images < 1.499)
        return mask

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']
        #print('clothsize', config['ClothSize'])

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos.flatten())
        pyflex.step()
    
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
    
    def get_flatten_positions(self):
        pos = self._flatten_pos()
        return pos.reshape(-1, 4)[:, :3].copy()
    
    def get_flatten_coverage(self):
        return self.get_coverage(self.get_flatten_positions())

   
    def _set_to_flatten(self):
        new_pos = self._flatten_pos()
        pyflex.set_positions(new_pos.flatten())
        self._target_img = self._get_obs()['image']
        self._flatten_corner_positions = self.get_corner_positions()
        new_pos = self.get_particle_positions()
        return self.get_coverage(new_pos)


    def get_visibility(self, positions=None, resolution=(64, 64)):
        # TODO: need to refactor this, so bad.
        # This has to be online.

        if positions is None:
            positions = self.get_particle_positions()

        N = positions.shape[0]
        
        visibility = [False for _ in range(N)]

        camera_hight = 1.5 # TODO: magic number
        depths = camera_hight - positions[:, 1] #x, z, y
        pixel_to_world_ratio = self.pixel_to_world_ratio # TODO: magic number
        projected_pixel_positions_x = positions[:, 0]/pixel_to_world_ratio/depths #-1, 1
        projected_pixel_positions_y = positions[:, 2]/pixel_to_world_ratio/depths #-1, 1
        projected_pixel_positions = np.concatenate(
            [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
            axis=1)

        depth_images = self.render(mode='rgbd')[:, :, 3]
        depth_images = cv2.resize(depth_images, resolution)
        #print(depth_images.shape)

        for i in range(N):
            x, y = projected_pixel_positions[i][0],  projected_pixel_positions[i][1]
            if x < -1 or x > 1 or y < -1 or y > 1:
                continue
            x_ = int((y + 1)/2 * resolution[0])
            y_ = int((x + 1)/2 * resolution[1])
            if depths[i] < depth_images[x_][y_] + 1e-6:
                visibility[i] = True
            
        return np.asarray(visibility), projected_pixel_positions

    def get_normalised_coverage(self):
        return self._normalised_coverage()
    
    def get_wrinkle_ratio(self):
        return self._get_wrinkle_pixel_ratio()
    
    def _get_wrinkle_pixel_ratio(self, particles=None):
        if particles is not None:
            old_particles = pyflex.get_positions()
            to_set_particles = old_particles.copy().reshape(-1, 4)
            to_set_particles[:, :3] = particles
            pyflex.set_positions(to_set_particles.flatten())
            #pyflex.step()

        rgb = self.render(mode='rgb')
        rgb = cv2.resize(rgb, (128, 128))
        mask = self.get_cloth_mask(pixel_size=(128, 128))

        if mask.dtype != np.uint8:  # Ensure mask has a valid data type (uint8)
            mask = mask.astype(np.uint8)

        # plt.imshow(mask)
        # plt.show()



        if particles is not None:
            pyflex.set_positions(old_particles)
            #pyflex.step()

        # Use cv2 edge detection to get the wrinkle ratio.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # plt.imshow(edges)
        # plt.show()

        masked_edges = cv2.bitwise_and(edges, mask)
        # plt.imshow(masked_edges)
        # plt.show()

        wrinkle_ratio = np.sum(masked_edges) / np.sum(mask)

        return wrinkle_ratio
    

    def get_corner_positions(self, position=None):
        if position == None:
            positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        return positions[self._corner_ids]
    
    def _normalised_coverage(self):
        return self._current_coverage_area/self._target_covered_area
    
    def get_particle_positions(self):
        pos = pyflex.get_positions()
        pos = pos.reshape(-1, 4)[:, :3].copy()
        return pos

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
        new_pos = np.empty(shape=(N, 4), dtype=float)
        new_pos[:, 0] = xx.flatten()
        new_pos[:, 1] = self.cloth_particle_radius
        new_pos[:, 2] = yy.flatten()
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        self._target_pos = new_pos.copy()

        return new_pos.copy()
    
    def get_normalised_coverage(self, particle_positions=None):
        if particle_positions is None:
            particle_positions = self.get_particle_positions()
        coverage_area = self.get_coverage(particle_positions) 
        return coverage_area/self.get_coverage(self.get_flatten_positions())
    
    def get_cloth_size(self):
        W, H =  self.get_current_config()['ClothSize']
        return H, W
    
    def get_coverage(self, pos):
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


    

    def _sample_cloth_size(self):
        return np.random.randint(60, 120), np.random.randint(60, 120)

    

    def get_camera_params(self):
        config = self.get_current_config()
        camera_name = config['camera_name']
        cam_pos = config['camera_params'][camera_name]['pos']
        cam_angle = config['camera_params'][camera_name]['angle']
        return cam_pos, cam_angle

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """

        cam_pos, cam_angle = np.array([-0.0, 1.5, 0]), np.array([0, -90 / 180. * np.pi, 0.])

        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [int(0.4 / self.cloth_particle_radius), int(0.4 / self.cloth_particle_radius)],
            'ClothStiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
            'front_colour':  [0.673, 0.111, 0.0],
            'back_colour': [0.612, 0.194, 0.394]
        }

        return config

    def get_step_info(self):
        if self.save_step_info:
            return self.step_info.copy()
        else:
            raise NotImplementedError

    def _get_obs(self):
        obs = {}
        if self.observation_mode['image'] == 'cam_rgb':
            obs['image'] = self.get_image(self.camera_height, self.camera_width)
        
        elif self.observation_mode['image'] == 'cam_rgbd':
            obs['image'] = self.get_image(self.camera_height, self.camera_width, depth=True)
        elif self.observation_mode['image'] == 'cam_d':
            obs['image'] = self.get_image(self.camera_height, self.camera_width, depth=True)[:, :, 3:4]
        else:
            raise NotImplementedError
        
        if 'state' not in self.observation_mode.keys():
            pass
        elif self.observation_mode['state'] == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            obs['state'] = pos
            

        elif self.observation_mode['state'] == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos
            obs['state'] = pos
        
        elif self.observation_mode['state'] == 'corner_pixel':
            positions =  self.get_corner_positions()
            N = positions.shape[0]
            camera_hight = 1.5 # TODO: magic number
            depths = camera_hight - positions[:, 1] #x, z, y
             # TODO: magic number

            projected_pixel_positions_x = positions[:, 0]/self.pixel_to_world_ratio/depths #-1, 1
            projected_pixel_positions_y = positions[:, 2]/self.pixel_to_world_ratio/depths #-1, 1
            projected_pixel_positions = np.concatenate(
                [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
                axis=1)
            
            obs['state'] = projected_pixel_positions.flatten()
        
        return obs

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 0 if 'env_idx' not in config else config['env_idx']
        mass = config['mass'] if 'mass' in config else 0.5
        

        if self.recolour_config:
            front_colour = self.context_random_state.uniform(
                np.array(self.context['colour']['front_colour']['lower_bound']), 
                np.array(self.context['colour']['front_colour']['upper_bound']))
            back_colour = self.context_random_state.uniform(
                np.array(self.context['colour']['back_colour']['lower_bound']),
                np.array(self.context['colour']['back_colour']['upper_bound']))

        else:
            front_colour = [0.673, 0.111, 0.0] if 'front_colour' not in config else config['front_colour']
            back_colour = [0.612, 0.194, 0.394] if 'back_colour' not in config else config['back_colour']


        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], mass,
                                 config['flip_mesh'], *front_colour, *back_colour])
        
        
        pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)
    
    def tick_control_step(self):
        super().tick_control_step()
        if self.save_step_info:
            self.step_info['rgbd'].append(self.get_image(height=self.save_image_dim[0], width=self.save_image_dim[1], depth=True)) #TODO: magic numbers
            self.step_info['coverage'].append(self.get_coverage(self.get_particle_positions()))
            self.step_info['reward'].append(self.compute_reward(self.get_particle_positions()))
            
            eval_data = self.evaluate()
            for k, v in eval_data.items():
                if k not in self.step_info.keys():
                    self.step_info[k] = [v]
                else:
                    self.step_info[k].append(v)

    def get_edge_ids(self):
        config = self.get_current_config()
        cloth_dimx,  cloth_dimy = config['ClothSize']
        edge_ids = [i for i in range(cloth_dimx)]
        edge_ids.extend([i*cloth_dimx for i in range(1, cloth_dimy)])
        edge_ids.extend([(i+1)*cloth_dimx-1 for i in range(1, cloth_dimy)])
        edge_ids.extend([(cloth_dimy-1)*cloth_dimx + i for i in range(1, cloth_dimx-1)])
        return edge_ids
    
    def _wait_to_stabalise(self, max_wait_step=300, stable_vel_threshold=0.0006,
            target_point=None, target_pos=None):
        t = 0
        stable_step = 0
        #print('stable vel threshold', stable_vel_threshold)
        last_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        for j in range(0, max_wait_step):
            t += 1

           
            cur_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            curr_vel = np.linalg.norm(cur_pos - last_pos, axis=1)
            if target_point != None:
                cur_poss = pyflex.get_positions()
                curr_vell = pyflex.get_velocities()
                cur_poss[target_point * 4: target_point * 4 + 3] = target_pos
                
                curr_vell[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(cur_poss.flatten())
                pyflex.set_velocities(curr_vell)
                curr_vel = curr_vell


            # curr_vel = pyflex.get_velocities()
            #curr_vel = pyflex.get_accelerations()
            #print('cur vel shape', curr_vel.shape)
            # if target_point != None:
            #     curr_pos = pyflex.get_positions()
            #     curr_pos[target_point * 4: target_point * 4 + 3] = target_pos
            #     curr_vel[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
            #     pyflex.set_positions(curr_pos)
            #     pyflex.set_velocities(curr_vel)

            self.tick_control_step()
            if stable_step > 10:
                break
            if np.max(curr_vel) < stable_vel_threshold:
                stable_step += 1
            else:
                stable_step = 0

            last_pos = cur_pos
            
        #print('wait steps', t)
        return t