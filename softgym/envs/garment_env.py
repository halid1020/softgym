import os
import sys
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2
import json

import pyflex

from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.qhull import QhullError
import matplotlib.pyplot as plt

## Create a enum from garment type to garment id
garment_type_to_id = {
    'Tshirt': 0,
    'Trousers': 1,
    'Dress': 2, 
    'Top': 3,
    'Jumpsuit': 4
}

class GarmentEnv(ClothEnv):
    def __init__(self,  **kwargs):       
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(kwargs['cached_states_path'], num_variations=kwargs['num_variations'])


    def get_default_config(self):
        #cam_pos, cam_angle = np.array([-0.0, 1.5, 0]), np.array([0, -90 / 180. * np.pi, 0.])
        config = {
            'pos': [0, 0, 0],
            'scale': 1.0,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': [1.2, 0.6, 1], #[0.8, 1, 0.9], #[0.85, 1, 0.9],
            'mass': 0.0003, # 0.005,
            'garment_type': 'Tshirt',
            'shape_id': 0,
            'radius': 0.005, #self.cloth_particle_radius,
            'camera_name': self.current_camera,
            'current_camera': self.current_camera,
            'camera_params': self.camera_params,
            'drop_height': 0.0,
            'front_colour':  [0.673, 0.111, 0.0],
            'back_colour': [0.673, 0.111, 0.0],
            'inside_colour':  [0.673, 0.111, 0.0],
        }

        return config
    
    def generate_env_variation(self, num_variations=1): #, vary_cloth_size=False)
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.02  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config().copy()

       

        for i in tqdm(range(num_variations), 'Generating Initial States'):
            config, state = self._generate_env_config(i)
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            #generated_configs[-1]['flatten_area'] = flatten_area  # Record the maximum flatten area

            #print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states
    
    def _generate_env_config(self, index):
        config = deepcopy(self.get_default_config().copy())
        self.random_state = np.random.RandomState(index)

        config = deepcopy(self.get_default_config().copy())
        if 'size' in self.context:
            config['scale'] = self.random_state.uniform(
                np.array(self.context['size']['scale']['lower_bound']),
                np.array(self.context['size']['scale']['upper_bound']))
        
        if 'colour' in self.context:
            config['front_colour'] = self.random_state.uniform(
                np.array(self.context['colour']['front_colour']['lower_bound']), 
                np.array(self.context['colour']['front_colour']['upper_bound']))
            config['back_colour'] = self.random_state.uniform(
                np.array(self.context['colour']['back_colour']['lower_bound']),
                np.array(self.context['colour']['back_colour']['upper_bound']))
            config['inside_colour'] = self.random_state.uniform(
                np.array(self.context['colour']['inside_colour']['lower_bound']),
                np.array(self.context['colour']['inside_colour']['upper_bound']))
            
        if 'garment' in self.context:
            print('vary garment type')
            #### choose a garment type from self.context['garment_type']
            config['garment_type'] = self.random_state.choice(self.context['garment'])


        #### The shapes are saved in PyFlex/data/TriGarments/<garment_type> folders
        ### PyFlex can be find in the environment variable PYFLEXROOT
        #### Get the number of shapes for the chosen garment type            
        num_shapes = len(os.listdir(os.path.join(os.environ['PYFLEXROOT'], 'data', 'TriGarments', config['garment_type'])))

        
        if 'shape' in self.context:
            print('vary shape')
            config['shape_id'] = int(self.random_state.uniform(
                self.context['shape']['lower_bound'],
                self.context['shape']['upper_bound'])*num_shapes)
        

        
            
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
        # if vary_cloth_size:
        #     cloth_dimx, cloth_dimy = self._sample_cloth_size()
        #     config['ClothSize'] = [cloth_dimx, cloth_dimy]
        # else:
        self.set_scene(config)
        # self.action_tool.reset([0., -1., 0.])
        flatten_area = self._set_to_flatten()
        self._wait_to_stabalise()

        if index == 0:
            return config, self.get_state()

        pos = pyflex.get_positions().reshape(-1, 4)
        pos[:, :3] -= np.mean(pos, axis=0)[:3]
        pos[:, 3] = 1
        pyflex.set_positions(pos.flatten())
        pyflex.set_velocities(np.zeros_like(pos))
        pyflex.step()

        num_particle = len(pos)
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

            center_object(self.random_state, self.context['position'])

            # Drag the cloth and wait to stablise
            if self.random_state.random() < 0.7:
                pickpoint = self.random_state.randint(0, num_particle - 1)
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

                center_object(self.random_state, self.context['position'])


        while True:
            center_object(self.random_state, self.context['position'])
            if self.context['rotation']:
                angle = self.random_state.rand(1) * np.pi * 2
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

            if np.alltrue(np.abs(projected_pixel_positions) < 1):
                break

        
        pyflex.step()
        pyflex.render()
        
        return config, self.get_state()
            
                
            

    def _set_to_flatten(self):
        pyflex.set_positions(self.canonical_pos)
        self._wait_to_stabalise()
        self._canonical_mask = self.get_cloth_mask()
        self.flatten_obs = self._get_obs()
        return self.get_coverage(self.get_particle_positions())
    
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
            render_mode = 2
        elif self.render_mode == 'cloth':
            render_mode = 1
        elif self.render_mode == 'both':
            render_mode = 3

        #print('cofig: ', config)

        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 5

        #print(config)

        if self.recolour_config:
            front_colour = random_state.uniform(
                np.array(self.context['colour']['front_colour']['lower_bound']), 
                np.array(self.context['colour']['front_colour']['upper_bound']))
            back_colour = random_state.uniform(
                np.array(self.context['colour']['back_colour']['lower_bound']),
                np.array(self.context['colour']['back_colour']['upper_bound']))
            
            inside_colour = random_state.uniform(
                np.array(self.context['colour']['inside_colour']['lower_bound']),
                np.array(self.context['colour']['inside_colour']['upper_bound']))

        else:
            front_colour = [0,0, 1.0, 1.0] if 'front_colour' not in config else config['front_colour']
            back_colour =   [0,0, 1.0, 1.0] if 'back_colour' not in config else config['back_colour']
            inside_colour =   [1.0, 0.0, 0.0] if 'inside_colour' not in config else config['inside_colour']

        print('config', config)
        self.current_config = config
        scene_params = np.concatenate([
            config['pos'][:], 
            [config['scale'], config['rot']], 
            config['vel'][:], 
            config['stiff'], 
            [config['mass'], config['radius']],
            camera_params['pos'][:], 
            camera_params['angle'][:], 
            [camera_params['width'], camera_params['height']], 
            [render_mode],
            front_colour,
            back_colour ,
            inside_colour,
            [garment_type_to_id[config['garment_type']], config['shape_id']]
        ])
        self.camera_params['default_camera'] = camera_params

        pyflex.set_scene(env_idx, scene_params, 0)
        self.default_pos = pyflex.get_positions()

        ### Get key ids
        self.canonicalise() ### set the cannical position
        self._read_key_points(config['garment_type'], config['shape_id'])

         # ### Plot the rgb image with annotated key points
        resolution = (720, 720)
        rgb = self.render(mode='rgb')
        rgb = cv2.resize(rgb, resolution)
        key_positions = self.get_keypoint_positions()
        print('key_positions', key_positions.shape)
        key_visible_positions, key_projected_positions = self.get_visibility(
            key_positions, resolution=resolution,
            camera_height=self.current_config['camera_params']['default_camera']['pos'][1])

        
        for vis, pos in zip(key_visible_positions, key_projected_positions):
            if (np.max(np.abs(pos)) < 1.0):
                pos = (pos + 1.0) * resolution[0] / 2
                rgb = cv2.circle(rgb, (int(pos[0]), int(pos[1])), 8, (0, 0, 255), 2)


        # plt.imshow(rgb)
        # plt.show()
        # ####################################################

        ### Set to Flatten
        self._flatten_coverage = self._set_to_flatten()
        #print('flatten coverage', self._flattened_coverage)
        self.flatten_pos = pyflex.get_positions().copy()

        ### Set the state
        if state is not None:
            self.set_state(state)
        self.initial_state_pos = pyflex.get_positions()
        self._initial_coverage = self.get_coverage()
    
    def get_key_ids(self):
        return self.key_ids
    
    def get_keypoint_positions(self):
        return self.get_particle_positions()[self.key_ids]

    def get_flatten_keypoint_positions(self):

        return self.flatten_pos.reshape(-1, 4)[:, :3][self.key_ids].copy()


    def canonicalise(self):
        pyflex.set_positions(self.default_pos)
        self.rotate_particles([180,180,0])
        positions = pyflex.get_positions().reshape(-1, 4)
        positions[:, 1] -= min(positions[:, 1]) + 0.01
        pyflex.set_positions(positions.flatten())
        pyflex.step()
        center_object(self.random_state, 0)#
        pyflex.step()
        self.canonical_pos = pyflex.get_positions()
    
    def get_flatten_positions(self):
        return self.flatten_pos.reshape(-1, 4)[:, :3].copy()

    

    
    def _get_convex_hull(self, particles_2d):
        hull = ConvexHull(particles_2d)
        

        vertices = hull.vertices
        
        ### If two vertices are too close in particles_2d, remove one

        while True:

            vertices_to_remove = []
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if np.linalg.norm(particles_2d[vertices[i]] - particles_2d[vertices[j]]) < 0.025:
                        vertices_to_remove.append(j)
            if len(vertices_to_remove) == 0:
                break    
            vertices = np.delete(vertices, vertices_to_remove)
            break

        return vertices

    
    def _read_key_points(self, garment, shape_id):

        data_dir = os.path.join(os.environ['PYFLEXROOT'], 'data', 'TriGarments', garment)

        ## put shape_id in 4 digits 0001, 0002, 0003, ...

        shape_id = str(shape_id).zfill(4)
        data_dir = os.path.join(data_dir, '{}_{}.json'.format(garment, shape_id))
        
        with open(data_dir, 'r') as f:
            data = json.load(f)
            self.name2keypoint = data.copy()
            self.key_ids = []
            for k, v in data.items():
                self.key_ids.append(v)


        print('key points', self.key_ids)


    def get_name2keypoints(self):
        return self.name2keypoint

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
            self._set_to_flatten()
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
    
    # def _step(self, action):

    #     if self.save_control_step_info:
    #         self.step_info = {}

    #     self.control_step +=  self.action_tool.step(action)
        
    #     if self.save_control_step_info:
    #         self.step_info = self.action_tool.get_step_info()
            
    #         self.step_info['coverage'] = []
    #         self.step_info['reward'] = []
    #         steps = len(self.step_info['control_signal'])

    #         for i in range(steps):
    #             particle_positions = self.step_info['particle_pos'][i][:, :3]
                
    #             self.step_info['rgbd'][i] = cv2.resize(self.step_info['rgbd'][i], self.save_image_dim)
    #             self.step_info['reward'].append(self.compute_reward(particle_positions))
    #             self.step_info['coverage'].\
    #                 append(self.get_coverage(particle_positions))
                
    #             eval_data = self.evaluate(particle_positions)
    #             for k, v in eval_data.items():
    #                 if k not in self.step_info.keys():
    #                     self.step_info[k] = [v]
    #                 else:
    #                     self.step_info[k].append(v)