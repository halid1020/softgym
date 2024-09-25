import cv2
import numpy as np
from gym.spaces import Box

class ClothVelocityControlEnv():
    _name = 'SoftGymClothEnv'


    def __init__(self, kwargs):
        #self.hoirzon = kwargs['horizon']
        self.eval_para = {
            'eval_tiers': kwargs['eval_tiers'],
            'video_episodes': kwargs['video_episodes']
        }
        
        self.eval_params = [
            {
                'eid': eid,
                'tier': tier,
                'save_video': (eid in kwargs['video_episodes'])
            }

            for tier in kwargs['eval_tiers'] for eid in kwargs['eval_tiers'][tier]
        ]

        self.val_params = [
            {
                'eid': eid,
                'save_video': False
            }
            for eid in kwargs['val_episodes']
        ]
        
        self._env = None
        self.action_mode = 'velocity-grasp'
        

    def get_num_picker(self):
        return self._env.get_num_picker()
    
    def set_to_flatten(self):
        info = self._env.set_to_flatten()
        return self._process_info(info)

    def get_particle_positions(self):
        return self._env.get_particle_positions()

    def get_picker_pos(self):
        return self._env.get_picker_pos()

    def get_picker_position(self):
        return self._env.get_picker_position()

    def set_disp(self, flag):
        self._env.set_gui(flag)

    def get_action_space(self):
        return self._env.get_action_space()
    
    def observation_shape(self): ### If image: H*W*C
        raise NotImplementedError
    
    # def step(self, action):
    #     raise NotImplementedError
    
    def get_flatten_canonical_IoU(self):
        mask = self.get_cloth_mask(resolution=(128, 128)).reshape(128, 128)
        canonical_mask = self.get_canonical_mask(resolution=(128, 128)).reshape(128, 128)
        intersection = np.sum(np.logical_and(mask, canonical_mask))
        union = np.sum(np.logical_or(mask, canonical_mask))
        IoU1 = intersection/union

        # Rotate the canonical mask by 90 degrees
        canonical_mask = np.rot90(canonical_mask)
        intersection = np.sum(np.logical_and(mask, canonical_mask))
        union = np.sum(np.logical_or(mask, canonical_mask))
        IoU2 = intersection / union

        return max(IoU1, IoU2)
    
    def get_eval_configs(self):
        return self.eval_params
    
    def get_val_configs(self):
        return self.val_params
    
    def get_flattened_pos(self):
        return self._env.get_flattened_pos()

    
    def set_eval(self):
        self._train = False
        self._env.eval_flag = True
    
    def set_val(self):
        self.set_eval()

    def set_train(self):
        self._train = True
        self._env.eval_flag = False
    
    def get_num_episodes(self):
        return self._env.get_num_episodes()

    def get_mode(self):
        if self._train:
            return 'train'
        else:
            return 'eval'

    def clear_frames(self):
        self._env.reset_control_step_info()

    def get_frames(self):
        res = self._env.get_control_step_info()
        return res['rgb']

    def get_info(self, info_keys):
        info = {}
        if 'rgb' in info_keys:
            info['rgb'] = self.render(mode='rgb', resolution=self.observation_shape()['rgb'][:2])
        if 'depth' in info_keys:
            info['depth'] = self.render(mode='d', resolution=self.observation_shape()['depth'][:2])

        if 'mask' in info_keys:
            info['mask'] = self.get_cloth_mask(resolution=self.observation_shape()['rgb'][:2])
        
        if 'normalised_coverage' in info_keys:
            info['normalised_coverage'] = self.get_normalised_coverage()

        if 'control_signal' in info_keys:
            control_info =  self.get_control_step_info()
            if 'control_signal' in control_info.keys():
                info['control_signal'] = control_info['control_signal']

        if 'control_normalised_coverage' in info_keys:
            control_info =  self.get_control_step_info()
            if 'control_normalised_coverage' in control_info.keys():
                info['control_normalised_coverage'] = control_info['control_normalised_coverage']
        
        if 'control_rgb' in info_keys:
            control_info =  self.get_control_step_info()
            info['control_rgb'] = control_info['rgb']
        
        if 'control_depth' in info_keys:
            control_info =  self.get_control_step_info()
            info['control_depth'] = control_info['depth']
        
        return info

    def is_falttened(self):
        return self.get_normalised_coverage() > 0.99
    
    
    def get_coverage(self):
        return self._env.get_coverage()
    
    
    def get_initial_coverage(self):
        return self._env.get_initial_coverage()
    
    def get_flattened_coverage(self):
        return self._env.get_flattened_coverage()

    def get_cloth_mask(self, camera_name='default_camera', resolution=(64, 64)):
        return self._env.get_cloth_mask(camera_name=camera_name, resolution=resolution)

    def get_canonical_mask(self, resolution=(64, 64)):
        return self._env.get_canonical_mask(resolution=resolution)

    def get_cloth_size(self):
        return self._env.get_cloth_size()

    def set_save_control_step_info(self, flag):
        self._env.set_save_control_step_info(flag)


    def get_object_positions(self):

        return self._env.get_particle_positions()

    

    def get_state(self):
        state = {
            'particle_pos': self._env.get_particle_pos(),
            'picker_pos': self._env.get_picker_pos(),
            'control_step': self._env.control_step,
            'action_step': self._t
        }

        return state  

    def set_state(self, state, step=None):
        self._env.set_pos(state['particle_pos'], state['picker_pos'])
        self._t = state['action_step']
        self._env.control_step = state['control_step']

    

    def get_performance_value(self):
        return self._env.get_performance_value()
    
    def wait_until_stable(self):
        info = self._env.wait_until_stable()
        return self._process_info(info)
    
    def get_visibility(self, positions, cameras="default_camera", resolution=(128,128)):
        N = positions.shape[0]
        
        visibility = [False for _ in range(N)]

        camera_hight = self.camera_height
        depths = camera_hight - (positions[:, 1]+ self._env.cloth_particle_radius) #x, z, y

        self.pixel_to_world_ratio = self._env.pixel_to_world_ratio
        projected_pixel_positions_x = positions[:, 0]/(self.pixel_to_world_ratio*depths) #-1, 1
        projected_pixel_positions_y = positions[:, 2]/(self.pixel_to_world_ratio*depths) #-1, 1
        projected_pixel_positions = np.concatenate(
            [projected_pixel_positions_x.reshape(N, 1), projected_pixel_positions_y.reshape(N, 1)],
            axis=1)

        depth_images = self.render(mode='d', camera_name=cameras, resolution=resolution)

        for i in range(N):
            x, y = projected_pixel_positions[i][0],  projected_pixel_positions[i][1]
            
            ## if not a number, continue
            if np.isnan(x) or np.isnan(y):
                continue
            if x < -1 or x > 1 or y < -1 or y > 1:
                continue
            x_ = int((y + 1)/2 * resolution[0])
            y_ = int((x + 1)/2 * resolution[1])
            ## clip x_, y_ between 0 and resolution
            x_ = max(0, min(x_, resolution[0]-1))
            y_ = max(0, min(y_, resolution[1]-1))
            
            if depths[i] < depth_images[x_][y_] + 1e-4:
                visibility[i] = True

        
        return [np.asarray(visibility)], [projected_pixel_positions]

    def get_goal(self):
        goal_ = self._env._goal
        goal = {}
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        goal['rgb'] = cv2.resize(goal_['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        goal['depth'] = cv2.resize(goal_['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        
       
        return goal
        
    def reset(self, episode_config=None):
        #print('reset')
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        self._t = 0  # Reset internal timer
        
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        self.set_save_control_step_info(episode_config['save_video'])
        info = self._env.reset(episode_id=episode_config['eid'])
        self.episode_id = self._env.episode_id
        episode_config['eid'] = self.episode_id
        self.episode_config = episode_config.copy()
        #print('reset end')
        return self._process_info(info)
    
    def get_episode_config(self):
        return self.episode_config
    
    def get_episode_id(self):
        return self.episode_id
    
    

    def get_pointcloud(self):
        particle_pos = self.get_object_positions()
        visibility, proj_pos = self.get_visibility(particle_pos)
        visible_particle_pos = particle_pos[tuple(visibility)]
        return visible_particle_pos

    

    def get_keypoint_positions(self):
        return self._env.get_keypoint_positions()

    def step(self, action, process_info=True):
        
        self._t += 1
        #print('lowest action start')
        info = self._env.step(action)
        #print('lowest action end')
        if process_info:
            info = self._process_info(info)
        return info

    def _process_info(self, info):
        #print('here process')
        
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        info['observation']['rgb'] = cv2.resize(info['observation']['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['depth'] = cv2.resize(info['observation']['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['mask'] = self.get_cloth_mask(resolution=(H, W))
        if 'contour' in self.info_keys:
            info['observation']['contour'] = self.get_contour(resolution=(H, W))
        info['no_op'] = self.get_no_op()
        if 'cloth_size' in self.info_keys:
            info['cloth_size'] = self.get_cloth_size()
        info['normalised_coverage'] = self.get_normalised_coverage()
        if 'corner_positions' in self.info_keys:
            info['corner_positions'] = self.get_corner_positions()
        info['flatten_canonical_IoU'] = self.get_flatten_canonical_IoU()
        if 'corner_visibility' in self.info_keys:
            info['corner_visibility'], _ = self.get_visibility(info['corner_positions'])
            info['corner_visibility'] = info['corner_visibility'][0]
        info['pointcloud'] = self.get_pointcloud()
        return info
    


    def render(self, mode='rgb', resolution=(720, 720), camera_name='default_camera'):
        return self._env.render(
            mode = mode,
            camera_name=camera_name, 
            resolution=resolution)

    def close(self):
        self._env.close()

    def get_action_horizon(self):
        return self._env.action_horizon

    def get_flatten_corner_positions(self):
        return self._env.get_flatten_corner_positions()
    
    def get_flatten_edge_positions(self):
        return self._env.get_flatten_edge_positions()

    def get_flatten_positions(self):
        return self._env.get_flatten_positions()
    

    def get_normalised_coverage(self):
        return max(0, min(1, self._env.get_normalised_coverage()))
    
    def get_wrinkle_ratio(self):
        return self._env.get_wrinkle_ratio()
 
    @property
    def observation_space(self):
        if self.symbolic:
            return self._env.observation_space
        else:
            return Box(low=-np.inf.astype(np.float32), high=np.inf.astype(np.float32), 
                shape=(self.observation_shape()['image'][0], self.observation_shape()['image'][1], self.observation_space()[2]), dtype=np.float32)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self._symbolic else (self._image_c, self._image_dim, self._image_dim)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]
    
    def sample_random_action(self):
        return self._env.action_space.sample()

    def get_flatten_observation(self):
        obs = self._env.get_flatten_observation()
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        obs['rgb'] = cv2.resize(obs['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        obs['depth'] = cv2.resize(obs['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        obs['mask'] = self.get_cloth_mask(resolution=(H, W))
        return obs
    
    def get_no_op(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError