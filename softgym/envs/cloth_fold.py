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
        self.cloth_dim = kwargs['cloth_dim']
        self.reward_mode = kwargs['reward_mode']
        self.initial_state = kwargs['initial_state']
        
        
        self.context = kwargs['context']
            

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)



    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        self._set_to_flatten()
        self.set_scene(self.cached_configs[self.current_config_id], self.cached_init_states[self.current_config_id])
        
        # If initiail state is set to flatten and it allows uses cached initial states
        # the, the enviornment allow change the center position and orientation of the fabric.
        if self.initial_state == 'flatten' and self.use_cached_states:
            self._set_to_flatten()
            center_object(self.context_random_state, self.context['position'])
            if self.context['rotation']:
                angle = self.context_random_state.rand(1) * np.pi * 2
                self._rotate_particles(angle)


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
        

    

        pyflex.step()
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
                
                self.step_info['rgbd'][i] = cv2.resize(self.step_info['rgbd'][i], self.save_image_dim) # TODO: magic number
                self.step_info['reward'].append(self.compute_reward(particle_positions))
                self.step_info['coverage'].\
                    append(self.get_coverage(particle_positions))


        if self.action_mode == 'pickerpickplace':
            self.action_step += 1
            self._wait_to_stabalise()
        

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
            longest_distance = 0.4*(2**0.5) ##TODO: magic number
            l =  self._largest_particle_distance()
            return (longest_distance - l)/longest_distance
        
        else:
            raise NotImplementedError

    #TODO: evaluation cannot deal with the wrinkle situations.
    def evaluate(self, particles=None):
        return {
            # This evaluation only can be done in simulation.
            'mean_particle_distance': self._mean_particle_distance(particles),
            'largest_particle_distanceq': self._largest_particle_distance(particles),
            
            # This can be done in perception
            'largest_corner_distance': self._largest_corner_distance(particles),
            'mean_edge_distance': self._mean_edge_distance(particles),
            'largest_edge_distance': self._largest_edge_distance(particles)
        }
    
    def _get_distance(self, positions, group_a, group_b):
        if positions is None:
            position = self.get_particle_positions()
        cols = [0, 2]
        pos_group_a = position[np.ix_(group_a, cols)]
        pos_group_b = position[np.ix_(group_b, cols)]
        distance = np.linalg.norm(pos_group_a-pos_group_b, axis=1)
        #print('distance', distance.shape)
        return distance
    
    def _mean_edge_distance(self, particles=None):
        
        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            edge_distance.append(np.mean([distances[i] for i, p in enumerate(group_ab) if p in edge_ids]))
        
        return np.min(edge_distance)

    def _largest_edge_distance(self, particles=None):

        distances = []
        edge_ids = self.get_edge_ids()
        edge_distance = []
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            edge_distance.append(np.max([distances[i] for i, p in enumerate(group_ab) if p in edge_ids]))
        
        return np.min(edge_distance)


    def _largest_corner_distance(self, particles=None):
        distances = []
        corner_distance = []
        for group_a, group_b in self.fold_groups:
            group_ab = np.concatenate([group_a, group_b])
            group_ba = np.concatenate([group_b, group_a])
            distances = self._get_distance(particles, group_ab, group_ba)
            corner_distance.append(np.max([distances[i] for i, p in enumerate(group_ab) if p in self._corner_ids]))
            #print(corner_distance)

        return np.min(corner_distance)


    def _mean_particle_distance(self, particles=None):

        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.mean(self._get_distance(particles, group_a, group_b)))
        
        return np.min(distances)


    def _largest_particle_distance(self, particles=None):
        distances = []
        for group_a, group_b in self.fold_groups:
            distances.append(np.max(self._get_distance(particles, group_a, group_b)))
        
        return np.min(distances)