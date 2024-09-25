import numpy as np
import cv2
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import  Picker
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.gemo_utils import *
from tqdm import tqdm

class ClothEnv(FlexEnv):

    reserved_cloth_param = {
        'pos': [-1.6, 2.0, -0.8],
        'size': [0.4, 0.4],
        'stiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
    }

    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', 
        picker_radius=0.02, picker_threshold=0.007, particle_radius=0.00625, mass=0.5,
        cloth_param=reserved_cloth_param,
        motion_trajectory='normal',
         **kwargs):
        #print('kwargs', kwargs)
        
        self.cloth_particle_radius = particle_radius
        self.cloth_param = cloth_param
        self.mass = mass
        
        super().__init__(**kwargs)
        

        self.render_mode = render_mode
        self.action_mode = action_mode
        self.pixel_to_world_ratio = 0.427 #0.4135 #0.4135
       
        camera_param = self.camera_params[self.current_camera]
        self.camera_height = camera_param['pos'][1]
        self.update_camera(self.current_camera, camera_param)
        #print('camera height', self.camera_height)
        
        self.num_picker = num_picker

        # Context
        self.recolour_config = kwargs['recolour_config']
        self.context = kwargs['context']
        self.random_state = np.random.RandomState(kwargs['random_seed'])
        

        #assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb', 'cam_rgbd']
        assert action_mode in ['velocity_control', 'pickerpickplace', 'pickerpickplace1', 'sawyer', 'franka', 'picker_qpg']
        self.observation_mode = observation_mode

        # if action_mode == 'velocity_control':
        self.action_tool = Picker(
            num_picker, picker_radius=picker_radius, 
            particle_radius=particle_radius, picker_threshold=picker_threshold,
            picker_low=kwargs['picker_low'], picker_high=kwargs['picker_high'],
            grasp_mode=(kwargs['grasp_mode'] if 'grasp_mode' in kwargs.keys() else {'closest': 1.0}),
        )
        self.action_space = self.action_tool.action_space
        self.picker_radius = picker_radius
        self.picker_initial_pos = kwargs['picker_initial_pos']
        #logging.info('[sofgym, cloth_env] picker_initial_pos: {}'.format(self.picker_initial_pos))
        

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
            self.state_space = Box(np.array([-np.inf] * sts_dim), np.array([np.inf] * sts_dim), dtype=np.float64)

        # if observation_mode['image'] == 'cam_rgb':
        #     self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
        #                                  dtype=np.float32)
                                    
        # elif observation_mode['image'] == 'cam_rgbd':
        #     self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 4),
        #                                  dtype=np.float32)

        # elif observation_mode['image'] == 'cam_d':
        #     self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 1),
        #                                  dtype=np.float32)
            
    def get_num_picker(self):
        return self.num_picker
            
    def get_action_space(self):
        return self.action_space
    
    def get_flatten_observation(self):
        return self.flatten_obs
    
    def get_keypoint_positions(self, particle_positions=None):
        raise NotImplementedError
    
    def get_edge_positions(self, particle_positions=None):
        if particle_positions is None:
            particle_positions = self.get_particle_positions()
        return particle_positions[self.get_edge_ids()]
    
    def get_corner_ids(self):
        return self._corner_ids
    
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
            
    def generate_env_variation(self, num_variations=1): #, vary_cloth_size=False)
        """ Generate initial states. Note: This will also change the current states! """
        
        generated_configs, generated_states = [], []
        

        for i in tqdm(range(num_variations), desc='Generating env variations'):

            config, state = self._generate_env_config(i)
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(state))
            self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = self._set_to_flatten()  # Record the maximum flatten area

            #print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

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

   

    def get_picker_pos(self):
        return self.action_tool.get_picker_pos()
    
    def get_picker_position(self):
        pos = self.get_picker_pos()
        return pos[:, :3].copy()

    

    def get_flatten_corner_positions(self):
        return self._flatten_corner_positions
    
    def get_flatten_edge_positions(self):
        return self._flatten_edge_positions

    def get_initial_coverage(self):
        return self._initial_coverage
    
    def get_canonical_mask(self, resolution=(64, 64)):
        #ret_mask = cv2.resize(self._canonical_mask.astype(np.float), resolution, interpolation=cv2.INTER_LINEAR)
        return self._canonical_mask

    def get_cloth_mask(self, camera_name='default_camera', resolution=(128, 128)):
        
        # This only works for top-down camera

        depth_images = self.render(camera_name=camera_name, mode='d')
        depth_images = cv2.resize(depth_images, resolution, interpolation=cv2.INTER_LINEAR)

        # TODO: make this better.
        mask = (self.camera_height - 0.15 < depth_images) & (depth_images < self.camera_height-0.001)

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
        self._canonical_mask = self.get_cloth_mask(camera_name=self.current_camera)
    
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
        new_pos[:, 0] = xx.flatten() * 0.95 #TODO
        new_pos[:, 1] = self.cloth_particle_radius
        new_pos[:, 2] = yy.flatten() * 0.95 #TODO
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        self._target_pos = new_pos.copy()

        return new_pos.copy()
    
    def get_flatten_positions(self):
        pos = self._flatten_pos()
        return pos.reshape(-1, 4)[:, :3].copy()
    
    def get_flattened_pos(self):
        return self._flatten_pos()
    
    
    def get_flattened_coverage(self):
        return self._flatten_coverage
    
    def set_to_flatten(self):
        self._set_to_flatten()
        obs =  self._get_obs()
        return {
            'observation': obs,
            'done': False
        }


    def get_visibility(self, positions=None, resolution=(64, 64), camera_height=None):
        # TODO: need to refactor this, so bad.
        # This has to be online.

        if positions is None:
            positions = self.get_particle_positions()

        N = positions.shape[0]
        
        visibility = [False for _ in range(N)]

        # camera_hight = 1.5 # TODO: magic number
        depths = camera_height - positions[:, 1] #x, z, y
        pixel_to_world_ratio = self.pixel_to_world_ratio 
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

    # def get_normalised_coverage(self):
    #     return self._normalised_coverage()
    

    
    
    
    def _normalised_coverage(self):
        return self._current_coverage_area/self._target_covered_area
    
    # def get_particle_positions(self):
    #     pos = pyflex.get_positions()
    #     pos = pos.reshape(-1, 4)[:, :3].copy()
    #     return pos

    def get_particle_positions(self):
        return self.get_particle_pos()[:, :3].copy()

    
    def get_normalised_coverage(self, particle_positions=None):
        coverage_area = self.get_coverage(particle_positions)
        #print('coverage area', coverage_area)
        return coverage_area/self._flatten_coverage
    
    def get_cloth_size(self):
        W, H =  self.get_current_config()['ClothSize']
        return H, W
    
    def get_coverage(self, positions=None):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        #pos = np.reshape(pos, [-1, 4])
        if positions is None:
            positions = self.get_particle_positions()

        min_x = np.min(positions[:, 0])
        min_y = np.min(positions[:, 2])
        max_x = np.max(positions[:, 0])
        max_y = np.max(positions[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = positions[:, [0, 2]]

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
        cam_pos = config['camera_param']['pos']
        cam_angle = config['camera_params']['angle']
        return cam_pos, cam_angle

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """

        config = {
            'ClothPos': self.cloth_param['pos'],
            'ClothSize': [int(self.cloth_param['size'][0]/ self.cloth_particle_radius), 
                          int(self.cloth_param['size'][1] / self.cloth_particle_radius)],

            'ClothStiff': self.cloth_param['stiff'],  # Stretch, Bend and Shear
            'camera_name': self.current_camera,
            'current_camera': self.current_camera,
            'camera_params': self.camera_params,
            'flip_mesh': 0,
            'front_colour':  [0.673, 0.111, 0.0],
            'back_colour': [0.612, 0.194, 0.394],
            'mass': self.mass
        }

        return config

    def get_step_info(self):
        if self.save_control_step_info:
            return self.control_step_info.copy()
        else:
            raise NotImplementedError
    
    def reset_control_step_info(self):
        super().reset_control_step_info()
        # if self.save_control_step_info:
        #     self.control_step_info['control_normalised_coverage'] = []

    def _get_obs(self):
        obs = {}
        obs['rgb'] = self.render(mode='rgb')
        
        obs['depth'] = self.render(mode='d')
        

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
           
            depths = self.camera_height - positions[:, 1] #x, z, y
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
        #print('config keys', config.keys())

        # This is temperorality like this
        camera_param = config['camera_params'][config['camera_name']]


        env_idx = 0 if 'env_idx' not in config else config['env_idx']
        mass = config['mass'] if 'mass' in config else 0.5
        #print('config', config)
        
        front_colour = [0.673, 0.111, 0.0] if 'front_colour' not in config else config['front_colour']
        back_colour = [0.612, 0.194, 0.394] if 'back_colour' not in config else config['back_colour']


        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_param['pos'][:], *camera_param['angle'][:], camera_param['width'], 
                                 camera_param['height'], mass,
                                 config['flip_mesh'], *front_colour, *back_colour])
        
        
        pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

        self._initial_coverage = self.get_coverage()
    
    def tick_control_step(self):
        super().tick_control_step()
        # if self.save_control_step_info:
        #     self.control_step_info['control_normalised_coverage'].append(self.get_normalised_coverage())


    def reset(self, episode_id=None):
       
        info = super().reset(episode_id=episode_id)
        # if self.save_control_step_info:
        #     self.control_step_info['control_normalised_coverage'] = [self.get_normalised_coverage()]
        return info

    
    
    def _step(self, action):
        #logging.debug(['[softgym, cloth_env] action', action])
        self.control_step +=  self.action_tool.step(action)
        self.tick_control_step()
        # if self.save_control_step_info:
        #     if 'control_signal' not in self.control_step_info:
        #         self.control_step_info['control_signal'] = []
        #     self.control_step_info['control_signal'].append(action)
    
    def wait_until_stable(self, max_wait_step=200, stable_vel_threshold=0.0006):
        wait_steps = self._wait_to_stabalise(max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold)
        # print('wait steps', wait_steps)
        obs = self._get_obs()

        done = False
        if (self.control_horizon is not None) and (self.control_step >= self.control_horizon):
            done = True


        return {
            'observation': obs,
            'done': done,
            'wait_steps': wait_steps
        }
    
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

            self.tick_control_step()
            # if self.save_control_step_info:
            #     if 'control_signal' not in self.control_step_info:
            #         self.control_step_info['control_signal'] = []
            #     self.control_step_info['control_signal'].append(np.zeros((self.num_picker, 4)))
                
            if stable_step > 10:
                break
            if np.max(curr_vel) < stable_vel_threshold:
                stable_step += 1
            else:
                stable_step = 0

            last_pos = cur_pos
            
        #print('wait steps', t)
        return t