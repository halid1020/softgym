import abc
import numpy as np
from gym.spaces import Box

import pyflex
import scipy.spatial
from enum import Enum

render_height, render_width = 720, 720

class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ Step funciton to change the action space states. Does not call pyflex.step() """


class Picker(ActionToolBase):
    class Status(Enum):
        PICK = 0
        HOLD = 1
        PLACE = 2
        

    def __init__(self, num_picker=1, picker_radius=0.05, init_pos=(0., -0.1, 0.), 
        picker_threshold=0.005, particle_radius=0.05, picker_low=(-0.4, 0., -0.4), 
        picker_high=(0.4, 1.0, 0.4), init_particle_pos=None, spring_coef=1.2, save_step_info=False, render=False, **kwargs):
        
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()

        self.save_step_info=save_step_info
        self._render = render
        
        if self.save_step_info:
            self.step_info = {
                'control_signal': [],
                'particle_pos': [],
                'picker_pos': [],
                'rgbd': []
            }

        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

        space_low = np.array([-0.1, -0.1, -0.1, 0] * self.num_picker) * 0.1  # [dx, dy, dz, [0, 1]]
        space_high = np.array([0.1, 0.1, 0.1, 10] * self.num_picker) * 0.1
        self.action_space = Box(space_low, space_high, dtype=np.float32)

    def update_picker_boundary(self, picker_low, picker_high):
        self.picker_low, self.picker_high = np.array(picker_low).copy(), np.array(picker_high).copy()

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.
        center = np.array(self.picker_high + self.picker_low) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        #print('picker_low', self.picker_low)
        clipped_picker_pos = picker_pos.copy()
        for i in range(3):
            if i == 1:
                #print('low z, high z, picker radius, input_pos', self.picker_low[i], self.picker_high[i], self.picker_radius, picker_pos[i])
                clipped_picker_pos[i] = np.clip(picker_pos[i], self.picker_low[i], self.picker_high[i])
        return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        
        if self.save_step_info:
            self.clean_step_info()

        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] + self.picker_low[i]) / 2.
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # pyflex.step() # Remove this as having an additional step here may affect the cloth drop env
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

    # num_pickers * 14
    def get_picker_pos(self):
        return np.array(pyflex.get_shape_states()).reshape(-1, 14)

    # num_pickers * 14
    def set_picker_pos(self, picker_pos):
        pyflex.set_shape_states(picker_pos)

    # num_pickers * 3
    def get_picker_position(self):
        return self.get_picker_pos()[:, :3]

    # num_pickers * 3
    def set_picker_position(self, picker_position):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_position
        self.set_picker_pos(shape_states)

    def get_particle_pos(self):
        return np.array(pyflex.get_positions()).reshape(-1, 4)

    @staticmethod
    def _get_pos():
        """ Get the current pos of the pickers and the particles, along with the inverse mass of each particle """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    @staticmethod
    def set_picker_pos(picker_pos):
        """ Caution! Should only be called during the reset of the environment. Used only for cloth drop environment. """
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)

    
    def render(self, mode='rgb'):
        #pyflex.step()
        img, depth_img = pyflex.render()
        H, W = render_height, render_width
   
        img = img.reshape(H, W, 4)[::-1, :, :3]  # Need to reverse the height dimension
        depth_img = depth_img.reshape(H, W, 1)[::-1, :, :1]

        if mode == 'rgbd':
            return np.concatenate((img, depth_img), axis=2)
        elif mode == 'rgb':
            return img
        elif mode == 'd':
            return depth_img
        else:
            raise NotImplementedError

    def clean_step_info(self):
        if self.save_step_info:
            self.step_info = {k: [] for k in self.step_info.keys()}
        
    def get_step_info(self):
        if self.save_step_info:
            return self.step_info.copy()
        else:
            raise NotImplementedError

    def step(self, action):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one, for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        
        action = np.reshape(action, (-1, 4))

        grip_flag = (action[:, 3] < 0)
        realse_flag = (0 < action[:, 3] < 1)
        
        picker_pos, particle_pos = self._get_pos()
        new_picker_pos, new_particle_pos = picker_pos.copy(), particle_pos.copy()

        # Un-pick the particles
        # print('check pick id:', self.picked_particles, new_particle_pos.shape, self.particle_inv_mass.shape)
        for i in range(self.num_picker):
            if (realse_flag[i] or grip_flag[i]) and self.picked_particles[i] is not None:
                #print('release ...')
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]  # Revert the mass
                self.picked_particles[i] = None
        
        self._set_pos(new_picker_pos, new_particle_pos)

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = self._apply_picker_boundary(picker_pos[i, :] + action[i, :3])
            if realse_flag[i]:
                continue

            if grip_flag[i]:  # No particle is currently picked and thus need to select a particle to pick
                #print('Intent to pick .....')
                dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                idx_dists = np.hstack([np.arange(particle_pos.shape[0]).reshape((-1, 1)), dists.reshape((-1, 1))])
                

                mask = dists.flatten() <= self.picker_threshold + self.picker_radius + self.particle_radius
                idx_dists = idx_dists[mask, :].reshape((-1, 2))
                

                idx_dists = list(idx_dists)
                idx_dists.sort(key=lambda i: (i[1], i[0]))
                idx_dists = np.asarray(idx_dists)

                #print('idx_dists', idx_dists[:3])

                if idx_dists.shape[0] > 0:
                    pick_id, pick_dist = None, None
                    for j in range(idx_dists.shape[0]):
                        if idx_dists[j, 0] not in self.picked_particles and (pick_id is None or idx_dists[j, 1] < pick_dist):
                            pick_id = idx_dists[j, 0]
                            pick_dist = idx_dists[j, 1]
                            break
                    if pick_id is not None:
                        self.picked_particles[i] = int(pick_id)

            if self.picked_particles[i] is not None:
                #print('holding....')
                # TODO The position of the particle needs to be updated such that it is close to the picker particle
                new_particle_pos[self.picked_particles[i], :3] = particle_pos[self.picked_particles[i], :3] + new_picker_pos[i, :] - picker_pos[i,
                                                                                                                                        :]
                new_particle_pos[self.picked_particles[i], 3] = 0  # Set the mass to infinity

        # check for e.g., rope, the picker is not dragging the particles too far away that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            l = len(picked_particle_idices)
            for i in range(l):
                for j in range(i + 1, l):
                    init_distance = np.linalg.norm(self.init_particle_pos[picked_particle_idices[i], :3] -
                                                   self.init_particle_pos[picked_particle_idices[j], :3])
                    now_distance = np.linalg.norm(new_particle_pos[picked_particle_idices[i], :3] -
                                                  new_particle_pos[picked_particle_idices[j], :3])
                    if now_distance >= init_distance * self.spring_coef:  # if dragged too long, make the action has no effect; revert it
                        new_picker_pos[active_picker_indices[i], :] = picker_pos[active_picker_indices[i], :].copy()
                        new_picker_pos[active_picker_indices[j], :] = picker_pos[active_picker_indices[j], :].copy()
                        new_particle_pos[picked_particle_idices[i], :3] = particle_pos[picked_particle_idices[i], :3].copy()
                        new_particle_pos[picked_particle_idices[j], :3] = particle_pos[picked_particle_idices[j], :3].copy()

        self._set_pos(new_picker_pos, new_particle_pos)
        
        
        ## Update environment
        pyflex.step()
        if self._render:
            pyflex.render()
        
        
        ### Save information
        if self.save_step_info:
            self.step_info['control_signal'].append(action)
            self.step_info['picker_pos'].append(self.get_picker_pos())
            self.step_info['particle_pos'].append(self.get_particle_pos())
            self.step_info['rgbd'].append(self.render('rgbd'))

        return 1


        


class PickerPickPlace(Picker):
    """
    4D action space: target position + pick/place, 
    If place, place everything from the begining.
    If pick, pick everything on the way.
    """
    def __init__(self, num_picker, env=None, picker_low=None, picker_high=None, 
        step_mode='world_pick_or_place', motion_trajectory='normal', picker_radius=0.02, 
        save_step_info=False, **kwargs):

        super().__init__(num_picker=num_picker,
                         picker_low=picker_low,
                         picker_high=picker_high,
                         picker_radius=picker_radius,
                         save_step_info=save_step_info,
                         **kwargs)
        
        
        if step_mode == "pixel_pick_and_place":
            self._pixel_to_world_ratio = 0.415 # While depth=1
            self._picker_low = np.asarray(picker_low)
            self._picker_high = np.asarray(picker_high)

            self._pick_height = kwargs['pick_height']
            self._place_height = kwargs['place_height']
            self._camera_depth = kwargs['camera_depth']

            #print(self._camera_depth, picker_high)

            picker_low = [picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth,
                          picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth]

            picker_high = [picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth,
                           picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth]
              
        self._motion_trajectory = motion_trajectory
        if motion_trajectory == 'triangle':
            self._intermidiate_height = kwargs['intermidiate_height']

        
        picker_low, picker_high = list(picker_low), list(picker_high)

        self.action_space = Box(np.array([*picker_low] * self.num_picker),
                                np.array([*picker_high] * self.num_picker), dtype=np.float32)
        self.delta_move = 0.01 # maximum velociy 1cm/frame
        self.env = env
        self._step_mode = step_mode

    def _world_pick_or_place(self, action, render=False):
        total_steps = 0
        action = action.reshape(-1, 4)
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
        end_pos = np.vstack([self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]])

        # print('curr_pos', curr_pos)
        # print('end pos', end_pos)
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move)) 
        if num_step < 0.1:
            return total_steps
        delta = (end_pos - curr_pos) / num_step # Get average distance need to move every step
        norm_delta = np.linalg.norm(delta) # Get the magtitude of the average distance.

        for i in range(int(min(num_step, 300))):  # The maximum number of steps allowed for one pick and place
            
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)])) # Apply average distanec to the target

            total_steps += 1
            if self.env is not None and self.env.recording:
                self.env.video_frames.append(self.env.render(mode='rgb_array'))
            
        return total_steps

    def _world_pick_and_place(self, action, render=False):
        total_steps = 0
        release_signal = 0.5
        grip_signal = -0.5

        # aciton: Num_pick * 2 (pick and place) * 3
        action = action.reshape(-1, 2, 3)
        pick_height = action[:, 0, 1]
        place_height = action[:, 1, 1]

        if self._motion_trajectory == 'normal':            

            # Raise to certain height, while releasing
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = place_height

            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)


            # Go to pick position while releasing without changing the height
            go_to_pick_pos_action = action[:, 0, :].copy()
            go_to_pick_pos_action[:, 1] = place_height
            go_to_pick_pos_action = \
                np.concatenate([go_to_pick_pos_action, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_pick_pos_action, render)

            # Lower the height
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = pick_height
            lower_action = np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(lower_action, render)

            # Pick
            super().step(np.hstack([np.zeros((1, 3)), np.zeros((1,1))]))
            total_steps += 1


            # Raise the height
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
            curr_pos[:, 1] = place_height
            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)


            # got the place position
            go_to_place_pos_action = action[:, 1, :].copy()
            go_to_place_pos_action = \
                np.concatenate([go_to_place_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            
            total_steps += self._world_pick_or_place(go_to_place_pos_action, render)

            # place
            super().step(np.hstack([np.zeros((1, 3)), np.full((1,1), release_signal)]))
            total_steps += 1
            
            # Move a bit
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 0] += 0.2
            curr_pos[:, 2] += 0.2
            move_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(move_action, render)


            total_steps += 1

        elif self._motion_trajectory == 'triangle':

            # Raise to certain height, while releasing
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = self._intermidiate_height

            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)

            # Go to pick position while releasing without changing the height
            go_to_pick_pos_action = action[:, 0, :].copy()
            go_to_pick_pos_action[:, 1] = self._intermidiate_height
            go_to_pick_pos_action = \
                np.concatenate([go_to_pick_pos_action, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_pick_pos_action, render)

            # Lower the height
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = pick_height
            lower_action = np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(lower_action, render)

            # Pick
            super().step(np.hstack([np.zeros((1, 3)), np.zeros((1,1))]))
            total_steps += 1


            # Go and Raise the height to the intermidiate position directly    
            go_to_int_pos_action = (action[:, 0, :].copy() + action[:, 1, :].copy())/2
            go_to_int_pos_action[:, 1] = self._intermidiate_height
            go_to_int_pos_action = \
                np.concatenate([go_to_int_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_int_pos_action, render)

            # Go and lower the height to the plce position directl
            go_to_place_pos_action = action[:, 1, :].copy()
            go_to_place_pos_action[:, 1] = place_height
            go_to_place_pos_action = \
                np.concatenate([go_to_place_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_place_pos_action, render)

            # place
            super().step(np.hstack([np.zeros((1, 3)), np.full((1,1), release_signal)]))
            total_steps += 1

             # Move a bit
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 0] += 0.05
            curr_pos[:, 1] = 0.1
            #curr_pos[:, 2] += 0.05
            move_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(move_action, render)

        

        return total_steps

    def _pixel_pick_and_place(self, action, render=True):
        # Input is 4D, normalised pixel position, [-1, 1]
        # Calculate world pick and place
        action = action.reshape(-1, 2, 2)
        pick_height = self._pick_height
        place_height = self._place_height

        xs = action[:, :, 0]*self._pixel_to_world_ratio
        ys = action[:, :, 1]*self._pixel_to_world_ratio

        xs[:, 0] = xs[:, 0]*(self._camera_depth - pick_height)
        xs[:, 1] = xs[:, 1]*(self._camera_depth - place_height)

        ys[:, 0] = ys[:, 0] *(self._camera_depth - pick_height)
        ys[:, 1] = ys[:, 1] *(self._camera_depth - place_height)
        
        new_action = np.zeros((self.num_picker, 2, 3))
        new_action[:, :, 0] = xs
        new_action[:, :, 2] = ys

        pick_heights = np.full(self.num_picker, pick_height)
        place_heights = np.full(self.num_picker, place_height)
        new_action[:, 0, 1] = pick_heights # x, z, y
        new_action[:, 1, 1] = place_heights #x,z,y

        new_action = new_action.flatten()
        
        return self._world_pick_and_place(new_action, render=render)


    def step(self, action, render=True, mode=None):
        """
        action: Array of pick_num x 4. For each picker, the action should be [x, y, z, pick/drop]. The picker will then first pick/drop, and keep
        the pick/drop state while moving towards x, y, x.
        """
        if self.save_step_info:
            self.clean_step_info()

        mode = self._step_mode if mode == None else mode
        if mode == "world_pick_or_place":
            return self._world_pick_or_place(action, render=render)
        
        if mode == "world_pick_and_place":
            return self._world_pick_and_place(action, render=render)

        if mode == "pixel_pick_and_place":
            return self._pixel_pick_and_place(action, render=render)

   
    


    def get_model_action(self, action, picker_pos):
        """Input the action and return the action used for GNN model prediction"""
        action = action.reshape(-1, 4)
        curr_pos = picker_pos
        end_pos = np.vstack([self._apply_picker_boundary(picker_pos) for picker_pos in action[:, :3]])
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return [], curr_pos
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        model_actions = []
        for i in range(int(min(num_step, 300))):  # The maximum number of steps allowed for one pick and place
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            model_actions.append(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            curr_pos += delta
            if np.alltrue(dist < self.delta_move):
                break
        return model_actions, curr_pos
    
    def sample(self):
        if self._step_mode == "pixel_pick_and_place":
            low = self._picker_low.reshape(-1, 2, 2)
            high = self._picker_high.reshape(-1, 2, 2)

            ret = np.random.rand(self.num_picker, 2, 2) * (high-low) + low
            return ret.flatten()


        return super().sample()