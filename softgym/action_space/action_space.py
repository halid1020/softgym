import abc
import numpy as np
from gym.spaces import Box

import pyflex
from enum import Enum
from scipy.spatial.distance import cdist

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
    
    def get_action_space(self):
        return self.action_space

    def __init__(self, num_picker=1, picker_radius=0.05, init_pos=(0., -0.1, 0.), 
        picker_threshold=0.007, particle_radius=0.05, picker_low=(-1, 0., -1), 
        picker_high=(1, 1.0, 1), init_particle_pos=None, spring_coef=1.2, save_step_info=False, render=False, grasp_mode={'closest': 1.0}, **kwargs):
        
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        # logging.info('[softgym, picker]  picker threshold: {}'.format(picker_threshold))
        
        self.set_save_step_info(save_step_info)
        self._render = render
        

        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [[] for _ in range (self.num_picker)]
        self.picker_low, self.picker_high = np.array(list(picker_low)).astype(np.float32), np.array(list(picker_high)).astype(np.float32)
        self.grasp_mode = grasp_mode
        
        # logging.info('[softgym, picker] num picker: {}'.format(self.num_picker))
        # logging.info('[softgym, picker] picker low: {}'.format(self.picker_low))
        # logging.info('[softgym, picker] picker high: {}'.format(self.picker_high))
        
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

        space_low = np.array([-0.1, -0.1, -0.1, -10] * self.num_picker) * 0.1  # [dx, dy, dz, [-1, 1]]
        space_high = np.array([0.1, 0.1, 0.1, 10] * self.num_picker) * 0.1
        self.action_space = Box(space_low.astype(np.float32), 
                                space_high.astype(np.float32), dtype=np.float32)
    
    def set_save_step_info(self, save_step_info):
        self.save_step_info=save_step_info
        if self.save_step_info:
            self.step_info = {
                'control_signal': [],
                'particle_pos': [],
                'picker_pos': [],
                'rgbd': []
            }

    # def update_picker_boundary(self, picker_low, picker_high):
    #     self.picker_low, self.picker_high = np.array(picker_low).copy(), np.array(picker_high).copy()

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.
        center = np.array(self.picker_high + self.picker_low) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        #print('picker_low', self.picker_low)
        #print('picker pos', picker_pos, self.picker_low, self.picker_high)
        return np.clip(picker_pos, self.picker_low, self.picker_high)
        #print('clipper picker pos', clipped_picker_pos)
        # for i in range(3):
        #     if i == 1:
        #         print('low z, high z, picker radius, input_pos', self.picker_low[i], self.picker_high[i], self.picker_radius, picker_pos[i])
        #         clipped_picker_pos[i] = np.clip(picker_pos[i], self.picker_low[:, i], self.picker_high[:, i])
        return clipped_picker_pos

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[i, 0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[i, 1]
            z = center[i, 2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, picker_pos):
        
        if self.save_step_info:
            self.clean_step_info()

        # for i in (0, 2):
        #     offset = center[i] - (self.picker_high[:, i] + self.picker_low[:, i]) / 2.
        #     self.picker_low[:, i] += offset
        #     self.picker_high[:, i] += offset
        init_picker_poses = picker_pos #self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            #print('!!!!add sphere')
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [[] for _ in range (self.num_picker)]
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        #centered_picker_pos = self._get_centered_picker_pos(center)
        centered_picker_pos = init_picker_poses
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # pyflex.step() # Remove this as having an additional step here may affect the cloth drop env
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

        self.last_grasp_mode = ['realease' for _ in range(self.num_picker)]
        self.graps_try_step = [0 for _ in range(self.num_picker)]

    # num_pickers * 14
    def get_picker_pos(self):
        #print('get picker pos', pyflex.get_shape_states())
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
        #print('picker_pos', picker_pos)
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
        #print('grasp mode:', self.grasp_mode)
        action = np.reshape(action, (-1, 4))
        grip_flag = action[:, 3] < 0
        release_flag = (0 <= action[:, 3]) & (action[:, 3] <= 1)
        
        picker_pos, particle_pos = self._get_pos()
        new_picker_pos = self._apply_picker_boundary(picker_pos + action[:, :3])
        new_particle_pos = particle_pos.copy()

        # Release particles
        release_mask = np.zeros(self.num_picker, dtype=bool)
        for i in np.where(release_flag)[0]:
            if self.picked_particles[i]:
                release_mask[i] = True
                self.last_grasp_mode[i] = 'release'
                self.graps_try_step[i] = 0
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = []

        # Pick new particles
        pick_mask = grip_flag & (np.array([len(p) for p in self.picked_particles]) == 0)
        
        if np.any(pick_mask):
            pickers_to_pick = np.where(pick_mask)[0]
            dists = cdist(picker_pos[pickers_to_pick], particle_pos[:, :3])
            
            threshold = self.picker_threshold + self.picker_radius + self.particle_radius
            #print('threshold:', threshold)
            mask = dists <= threshold
            #print('grasp num', np.sum(mask, axis=1))
            
            for i, picker_idx in enumerate(pickers_to_pick):
                valid_particles = np.where(mask[i])[0]
                if len(valid_particles) > 0:
                    sorted_indices = np.argsort(dists[i, valid_particles])
                    candidate_particles = valid_particles[sorted_indices]
                    
                    mode = np.random.choice(list(self.grasp_mode.keys()), p=list(self.grasp_mode.values()))

                    if self.last_grasp_mode[i] == 'miss' and self.graps_try_step[i] < 40:
                        self.graps_try_step[i] += 1
                    elif mode == 'around':
                        self.picked_particles[picker_idx].extend(candidate_particles)
                        self.last_grasp_mode[i] = 'around'
                    elif mode == 'closest':
                        self.picked_particles[picker_idx].append(candidate_particles[0])
                        self.last_grasp_mode[i] = 'closest'
                    elif mode == 'miss':
                        self.last_grasp_mode[i] = 'miss'
                    else:
                        raise NotImplementedError
                    
                    #print('len picked particles:', len(self.picked_particles[picker_idx]))

                    # 'miss' mode: do nothing

        # Update picked particle positions
        for i, particles in enumerate(self.picked_particles):
            if particles:
                displacement = new_picker_pos[i] - picker_pos[i]
                new_particle_pos[particles, :3] += displacement
                new_particle_pos[particles, 3] = 0  # Set mass to infinity

        self._set_pos(new_picker_pos, new_particle_pos)
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
            #self._pixel_to_world_ratio = #0.4135 # TODO; magic number, While depth=1
            self._picker_low = np.asarray(picker_low)
            self._picker_high = np.asarray(picker_high)

            self._pick_height = kwargs['pick_height']
            self._place_height = kwargs['place_height']
            self._camera_depth = kwargs['camera_depth']
            self._end_trajectory_move = kwargs['end_trajectory_move']

            picker_low = self.picker_low
            picker_high = self.picker_high

            # world_picker_low = [picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth,
            #                picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth]

            # # picker_low = [picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth,
            # #               picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth]

            # world_picker_hight = [picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth,
            #                picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth]
        
        elif step_mode == "pixel_pick_and_place_z":
            #self._pixel_to_world_ratio = #0.4135 # TODO; magic number, While depth=1
            self._picker_low = np.asarray(picker_low)
            self._picker_high = np.asarray(picker_high)
            self._camera_depth = kwargs['camera_depth']
            self._end_trajectory_move = kwargs['end_trajectory_move']

            picker_low = self.picker_low
            picker_high = self.picker_high


            # picker_low = [picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth,
            #               picker_low[0]*self._pixel_to_world_ratio*self._camera_depth, 0, picker_low[1]*self._pixel_to_world_ratio*self._camera_depth]

            # picker_high = [picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth,
            #                picker_high[0]*self._pixel_to_world_ratio*self._camera_depth, self._camera_depth, picker_high[1]*self._pixel_to_world_ratio*self._camera_depth]
        else:
            raise NotImplementedError

        self._motion_trajectory = motion_trajectory
        self._intermidiate_height = kwargs['intermidiate_height']
        self._release_height = kwargs['release_height']
        if motion_trajectory == 'triangle':
            pass
        elif motion_trajectory == 'triangle_with_height_ratio':
            self._intermidiate_height_ratio = kwargs['intermidiate_height_ratio']
            self._minimum_intermidiate_height = kwargs['minimum_intermidiate_height']
            self._maximum_intermidiate_height = kwargs['maximum_intermidiate_height']

        
        picker_low, picker_high = list(picker_low), list(picker_high)

        self.action_space = Box(np.array(picker_low),
                                np.array(picker_high), dtype=np.float32)
        self.delta_move = 0.01 # maximum velociy 2cm/frame
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
            curr_pos[:, 1] = self._release_height

            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)


            # Go to pick position while releasing without changing the height
            go_to_pick_pos_action = action[:, 0, :].copy()
            go_to_pick_pos_action[:, 1] = self._release_height
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
            curr_pos[:, 1] = self._intermidiate_height
            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)


            # got the place position
            go_to_place_pos_action = action[:, 1, :].copy()
            go_to_place_pos_action = \
                np.concatenate([go_to_place_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            
            total_steps += self._world_pick_or_place(go_to_place_pos_action, render)

            # Lower the height
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = place_height
            lower_action = np.concatenate([curr_pos, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(lower_action, render)

            # place
            super().step(np.hstack([np.zeros((1, 3)), np.full((1,1), release_signal)]))
            total_steps += 1
            
            # Move a bit
            if self._end_trajectory_move:
                curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
                displacement = 0.04*np.sign(action[:, 1, :].copy() - action[:, 0, :].copy())
                curr_pos = curr_pos + displacement
                curr_pos[:, 1] = self._release_height
                move_action = \
                    np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
                total_steps += self._world_pick_or_place(move_action, render)


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
            if self._end_trajectory_move:
                curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
                displacement = 0.04*np.sign(action[:, 1, :].copy() - action[:, 0, :].copy())
                curr_pos = curr_pos + displacement
                curr_pos[:, 1] = self._release_height
                move_action = \
                    np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
                total_steps += self._world_pick_or_place(move_action, render)

        elif self._motion_trajectory == 'triangle_with_height_ratio':

            # Raise to certain height, while releasing
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].reshape(self.num_picker, -1)

            curr_pos[:, 1] = self._release_height

            

            raise_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(raise_action, render)

            # Go to pick position while releasing without changing the height
            go_to_pick_pos_action = action[:, 0, :].copy()
            go_to_pick_pos_action[:, 1] = self._release_height
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
            
            go_to_int_pos_action[:, 1] = min(max(
                self._intermidiate_height_ratio * np.linalg.norm(action[:, 1, :].copy() - action[:, 0, :].copy(), axis=1),
                self._minimum_intermidiate_height), self._maximum_intermidiate_height)
        
            go_to_int_pos_action = \
                np.concatenate([go_to_int_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_int_pos_action, render)

            # Go and lower the height to the plce position directl
            #print('place_height', place_height)
            go_to_place_pos_action = action[:, 1, :].copy()
            go_to_place_pos_action[:, 1] = place_height
            go_to_place_pos_action = \
                np.concatenate([go_to_place_pos_action, np.full((self.num_picker, 1), grip_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(go_to_place_pos_action, render)

            # place
            super().step(np.hstack([np.zeros((1, 3)), np.full((1,1), release_signal)]))
            total_steps += 1

            #Raise
            curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
            curr_pos[:, 1] = self._release_height
            move_action = \
                np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
            total_steps += self._world_pick_or_place(move_action, render)

            # Move a bit
            if self._end_trajectory_move:
                curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3].copy()
                displacement = 0.04*np.sign(action[:, 1, :].copy() - action[:, 0, :].copy())
                curr_pos = curr_pos + displacement
                curr_pos[:, 1] = self._release_height
                move_action = \
                    np.concatenate([curr_pos, np.full((self.num_picker, 1), release_signal)], axis=1).flatten()
                total_steps += self._world_pick_or_place(move_action, render)
        else:
            raise NotImplementedError

        

        return total_steps

    def _pixel_pick_and_place(self, action, render=True):
        # Input is 4D, normalised pixel position, [-1, 1]
        # Calculate world pick and place
        action = action.reshape(-1, 2, 2)
        new_action = np.zeros((self.num_picker, 2, 3))
        new_action[:, :, :2] = action
        new_action[:, 0, 2] = self._pick_height
        new_action[:, 1, 2] = self._place_height
        return self._pixle_pick_and_place_z(new_action, render)

    
    def _pixle_pick_and_place_z(self, action, render=True):
        action = action.reshape(-1, 2, 3)
        pick_height = action[:, 0, 2]
        place_height = action[:, 1, 2]

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
        
        if mode == 'pixel_pick_and_place_z':
            return self._pixle_pick_and_place_z(action, render=render)
        
        raise NotImplementedError

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
        else:
            raise NotImplementedError