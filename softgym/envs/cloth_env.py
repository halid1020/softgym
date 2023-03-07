import numpy as np
import cv2
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import  Picker, PickerPickPlace
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid

class ClothEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', 
        picker_radius=0.02, picker_threshold=0.002, particle_radius=0.00625, 
        motion_trajectory='normal',
         **kwargs):
        
        self.cloth_particle_radius = particle_radius
        
        super().__init__(**kwargs)

        self.render_mode = render_mode
        self.action_mode = action_mode
       
        

        #assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb', 'cam_rgbd']
        assert action_mode in ['picker', 'pickerpickplace', 'pickerpickplace1', 'sawyer', 'franka', 'picker_qpg']
        self.observation_mode = observation_mode

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius, picker_threshold=picker_threshold,
                                      picker_low=(-0.4, 0., -0.4), picker_high=(1.0, 0.5, 0.4))
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


        if observation_mode['state'] == None:
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
        
        if observation_mode['state'] != None:
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

    def get_particle_pos(self):
        pos = pyflex.get_positions()
        return pos.reshape(-1, 4).copy()

    def set_particle_pos(self, pos):
        pyflex.set_positions([pos])
        print('here')
        # pyflex.step()
        # if self._render:
        #     pyflex.render()

    def get_particle_positions(self):
        return  self.get_particle_pos()[:, :3].copy()

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
        pyflex.set_positions(curr_pos)
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

   
    def _set_to_flatten(self):
        new_pos = self._flatten_pos()
        pyflex.set_positions(new_pos.flatten())
        self._target_img = self._get_obs()['image']
        self._flatten_corner_positions = self.get_corner_positions()
        new_pos = self.get_particle_positions()
        return self.get_coverage(new_pos)


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

    def get_flatten_coverage(self):
        return self.get_coverage(self.get_flatten_positions())
    
    def get_normalised_coverage(self, particle_positions=None):
        if particle_positions is None:
            particle_positions = self.get_particle_positions()
        coverage_area = self.get_coverage(particle_positions) 
        return coverage_area/self.get_coverage(self.get_flatten_positions())
    
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
            'flip_mesh': 0
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
        
        if self.observation_mode['state'] == 'point_cloud':
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
            positions =  self._get_corner_positions()
            N = positions.shape[0]
            camera_hight = 1.5 # TODO: magic number
            depths = camera_hight - positions[:, 1] #x, z, y
            pixel_to_world_ratio = 0.415 # TODO: magic number

            projected_pixel_positions_x = positions[:, 0]/pixel_to_world_ratio/depths #-1, 1
            projected_pixel_positions_y = positions[:, 2]/pixel_to_world_ratio/depths #-1, 1
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
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], mass,
                                 config['flip_mesh']])
        
        
        pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)
    
    def tick_control_step(self):
        super().tick_control_step()
        if self.save_step_info:
            self.step_info['rgbd'].append(self.get_image(width=64, height=64, depth=True)) #TODO: magic numbers
            self.step_info['coverage'].append(self.get_coverage(self.get_particle_positions()))
            self.step_info['reward'].append(self.compute_reward(self.get_particle_positions()))

    
    def _wait_to_stabalise(self, max_wait_step=20, stable_vel_threshold=0.05, target_point=None, target_pos=None, render=False):
        t = 0
        for j in range(0, max_wait_step):
            curr_vel = pyflex.get_velocities()
            if target_point != None:
                curr_pos = pyflex.get_positions()
                curr_pos[target_point * 4: target_point * 4 + 3] = target_pos
                curr_vel[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            self.tick_control_step()
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                break
        
        return t