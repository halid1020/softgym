import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import cv2

import pyflex

from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object

class TshirtEnv(ClothEnv):
    def __init__(self,  **kwargs):       
        super().__init__(**kwargs)

        self.get_cached_configs_and_states("", num_variations=1)

        print('render_mode', kwargs['render_mode'])


        # #### Current Just get the default config
        # self.cached_configs = [self.get_default_config()]
        # self.set_scene(self.cached_configs[0])
        # self.cached_init_states = [self.get_state()]


    def get_default_config(self):
        cam_pos, cam_angle = np.array([-0.0, 1.5, 0]), np.array([0, -90 / 180. * np.pi, 0.])
        config = {
            'pos': [0, 0, 0],
            'scale': 0.3,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': [0.5, 1, 0.9],
            'mass': 0.05,
            'radius': self.cloth_particle_radius,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'drop_height': 0.0
        }

        return config
    
    def generate_env_variation(self, num_variations=1):
        generated_configs, generated_states = [], []
        default_config = self.get_default_config().copy()

        self.update_camera(default_config['camera_name'], default_config['camera_params'][default_config['camera_name']])
        self.set_scene(default_config)
        center_object(self.context_random_state, 0)

        generated_configs.append(deepcopy(default_config))
        generated_states.append(deepcopy(self.get_state()))

        print(generated_states[0]['camera_params'])

        return generated_configs, generated_states

    def _set_to_flatten(self):
        pyflex.set_positions(self.default_pos)
        self.rotate_particles([180,0,90])
        pyflex.step()

    def rotate_particles(self, angle):
        r = R.from_euler('zyx', angle, degrees=True)
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()[:,:3]
        new_pos = r.apply(new_pos)
        new_pos = np.column_stack([new_pos,pos[:,3]])
        new_pos += center
        pyflex.set_positions(new_pos)

    
    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3

        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 5
        scene_params = np.concatenate([
            config['pos'][:], 
            [config['scale'], config['rot']], 
            config['vel'][:], 
            config['stiff'], 
            [config['mass'], config['radius']],
            camera_params['pos'][:], 
            camera_params['angle'][:], 
            [camera_params['width'], camera_params['height']], 
            [render_mode]])
        pyflex.set_scene(env_idx, scene_params, 0)
        if state is not None:
            self.set_state(state)
        self.default_pos = pyflex.get_positions().reshape(-1, 4)

    def _reset(self):
        """ Right now only use one initial state"""
        
        self.set_scene(self.cached_configs[self.current_config_id], self.cached_init_states[self.current_config_id])
        self._set_to_flatten()
        self._wait_to_stabalise()
        # self._flatten_particel_positions = self.get_flatten_positions()
        # self._flatten_coverage =  self.get_coverage(self._flatten_particel_positions)
        
        self._initial_particel_positions = self.get_particle_positions()
        self._initial_coverage = self.get_coverage(self._initial_particel_positions)
    

        if self.action_mode == 'pickerpickplace':
            self.action_step = 0
            self._current_action_coverage = self._prior_action_coverage = self._initial_coverage


        if hasattr(self, 'action_tool'):
            self.action_tool.reset(np.asarray([0.2, 0.2, 0.2]))

        # if hasattr(self, 'action_tool'):
        #     particle_pos = pyflex.get_positions().reshape(-1, 4)
        #     p1, p2, p3, p4 = self._get_key_point_idx()
        #     key_point_pos = particle_pos[(p1, p2), :3]
        #     middle_point = np.mean(key_point_pos, axis=0)
        #     self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
            
        pyflex.step()
        self.init_covered_area = None
        return self._get_obs(), None
    
    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        return 1.0
    
    def evaluate(self):
        return {}
    
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
    

    # def reset(self,given_goal=None, given_goal_pos=None):
    #     self.set_scene()
    #     self.particle_num = pyflex.get_n_particles()
    #     self.prev_reward = 0.
    #     self.time_step = 0
    #     self._set_to_flatten()
    #     if hasattr(self, 'action_tool'):
    #         self.action_tool.reset([0, 0.1, 0])
    #         self.set_picker_pos(self.reset_pos)
    #     self.goal = given_goal 
    #     self.goal_pos = given_goal_pos 
    #     if self.recording:
    #         self.video_frames.append(self.render(mode='rgb_array'))

    #     self.render(mode='rgb_array')
    #     self._set_to_flatten()
    #     self.move_to_pos([0,0.05,0])
    #     for i in range(10):
    #         pyflex.step()
    #         #self.render(mode='rgb_array')
    #     obs = self._get_obs()

    #     return obs