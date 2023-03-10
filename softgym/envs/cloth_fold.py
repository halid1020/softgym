from turtle import shape
from matplotlib.pyplot import step
import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
import cv2

class ClothFoldEnv(ClothEnv):

    def __init__(self, cached_states_path='cloth_folding_tmp.pkl', **kwargs):
        self.cloth_particle_radius = kwargs['particle_radius']
        
        super().__init__(**kwargs)

        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        self.cloth_dim = kwargs['cloth_dim']
        self.fold_mode = kwargs['fold_mode']
        self.reward_mode = kwargs['reward_mode']
        self.initial_state = kwargs['initial_state']
        
        if self.use_cached_states == False:
            self.context = kwargs['context']
            self.context_random_state = np.random.RandomState(kwargs['random_seed'])

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = int(self.cloth_dim[0]/self.cloth_particle_radius), int(self.cloth_dim[1]/self.cloth_particle_radius)
                
                
                self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object(self.context_random_state, self.context['positions'])
            
            if self.context['rotations']:
                angle = self.context_random_state.rand(1) * np.pi * 2
                self._rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        self._set_to_flatten()
        self.set_scene(self.cached_configs[self.current_config_id], self.cached_init_states[self.current_config_id])
        
        self._flatten_particel_positions = self.get_flatten_positions()
        self._flatten_coverage =  self.get_coverage(self._flatten_particel_positions)
        
        self._initial_particel_positions = self.get_particle_positions()
        self._initial_coverage = self.get_coverage(self._initial_particel_positions)
    

        if self.action_mode == 'pickerpickplace':
            self.action_step = 0
            self._current_action_coverage = self._prior_action_coverage = self._initial_coverage
        
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3]
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])
        

        
        config = self.get_current_config()
        cloth_dimx, cloth_dimz = config['ClothSize']
        #print('cloth size', config['ClothSize'])
        self._corner_ids = [0, cloth_dimx-1, (cloth_dimz-1)*cloth_dimx, cloth_dimz*cloth_dimx-1]

        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][0], config['ClothSize'][1]).T  # Reversed index here
        vertical_flip_particle_grid_idx = np.flip(particle_grid_idx, 1)


        if self.fold_mode == 'diagonal': ### Only Valid For Square Fabrics
            upper_triangle_ids = np.triu_indices(cloth_dimx)
            self.fold_group_a = particle_grid_idx[upper_triangle_ids].flatten()
            self.fold_group_b = particle_grid_idx.T[upper_triangle_ids].flatten()
            self.fold_group_a_flip =  vertical_flip_particle_grid_idx[upper_triangle_ids].flatten()
            self.fold_group_b_flip = vertical_flip_particle_grid_idx.T[upper_triangle_ids].flatten()

            
        elif self.fold_mode == 'side': ## Need to test this out.
            cloth_dimx = config['ClothSize'][0]
            x_split = cloth_dimx // 2
            self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
            self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        else:
            raise NotImplementedError

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1

        pyflex.step()

        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        # info = self._get_info()
        # self.performance_init = info['performance']
        return self._get_obs(), None

    def _step(self, action):

        self.control_step +=  self.action_tool.step(action)
        
        if self.save_step_info:
            self.step_info = self.action_tool.get_step_info()
            
            self.step_info['coverage'] = []
            self.step_info['reward'] = []
            steps = len(self.step_info['control_signal'])

            for i in range(steps):
                particle_positions = self.step_info['particle_pos'][i][:, :3]
                
                self.step_info['rgbd'][i] = cv2.resize(self.step_info['rgbd'][i], (64, 64)) # TODO: magic number
                self.step_info['reward'].append(self.compute_reward(particle_positions))
                self.step_info['coverage'].\
                    append(self.get_coverage(particle_positions))


        if self.action_mode == 'pickerpickplace':
            self.action_step += 1
            self._wait_to_stabalise(render=True,  max_wait_step=20, stable_vel_threshold=0.05)

        else:
            self.tick_control_step()

        if self.save_step_info:
            self.step_info = {k: np.stack(v) for k, v in self.step_info.items()}
        
        ### Update parameters for quasi-static pick and place.
        if self.action_mode == 'pickerpickplace':
            self._prior_action_coverage = self._current_action_coverage
            self._current_action_coverage = self.get_coverage(self.get_particle_positions())



    def compute_reward(self, particle_positions=None):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        if particle_positions is None:
            particle_positions = pyflex.get_positions()
            particle_positions = particle_positions.reshape((-1, 4))[:, :3]
        
        if self.reward_mode == 'normalised_particle_distance':
            longest_distance = 0.4*(2**0.5)
            l =  self._largest_particle_distance()
            return (longest_distance - l)/longest_distance
        
        else:
            raise NotImplementedError

    def evaluate(self, particles=None):
        return {
            # This evaluation only can be done in simulation.
            'mean_particle_distance': self._mean_particle_distance(particles),
            'largest_particle_disntace': self._largest_particle_distance(particles),
            
            # This can be done in perception
            'corner_distance': self._corner_distance(particles),
            'mean_edge_distance': self._mean_edge_distance(particles),
            'largest_edge_distance': self._largest_edge_distance(particles),
        }

    def _get_distance(self, positions, group_a, group_b):
        if positions is None:
            position = self.get_particle_positions()
        cols = [0, 1, 2]
        pos_group_a = position[np.ix_(group_a, cols)]
        pos_group_b = position[np.ix_(group_b, cols)]
        distance = np.linalg.norm(pos_group_a-pos_group_b, axis=1)
        return distance
    
    def _mean_edge_distance(self, particles=None):
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            edge_ids = self.get_edge_ids()
            edge_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in edge_ids]
            edge_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in edge_ids]

            return min(np.mean(edge_distance_1), np.mean(edge_distance_2))
        else:
            raise NotImplementedError

    def _largest_edge_distance(self, particles=None):
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            edge_ids = self.get_edge_ids()
            edge_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in edge_ids]
            edge_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in edge_ids]

            return min(np.max(edge_distance_1), np.max(edge_distance_2))
        else:
            raise NotImplementedError

    def _corner_distance(self, particles=None):

        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            corner_distance_1 = [distances_1[i] for i, p in enumerate(self.fold_group_a) if p in self._corner_ids]
            corner_distance_2 = [distances_2[i] for i, p in enumerate(self.fold_group_a_flip) if p in self._corner_ids]

            return min(np.max(corner_distance_1), np.max(corner_distance_2))
        else:
            raise NotImplementedError

    def _mean_particle_distance(self, particles=None):

        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(particles, self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(particles, self.fold_group_a_flip, self.fold_group_b_flip)
            return min(np.mean(distances_1), np.mean(distances_2))
        else:
            raise NotImplementedError



    def _largest_particle_distance(self):
        if self.fold_mode == 'diagonal':
            distances_1 = self._get_distance(self.fold_group_a, self.fold_group_b) ## particle-wise distane
            distances_2 = self._get_distance(self.fold_group_a_flip, self.fold_group_b_flip)
            return min(np.max(distances_1), np.max(distances_2))
        else:
            raise NotImplementedError


    def is_folded(self, particles=None):
        return self._largest_particle_distance(particles) < 0.05
