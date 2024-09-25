import numpy as np
from gym.spaces import Box

from .world_position_with_velocity_and_grasping_control_wrapper \
    import WorldPositionWithVelocityAndGraspingControlWrapper


class WorldPickAndPlaceWrapper():

    def __init__(self, 
                 env,
                 action_horizon=20,

                 velocity=0.1,
                 motion_trajectory='triangle_with_height_ratio',

                 pick_height=0.025,
                 place_height=0.06,


                 pick_lower_bound=[-1, -1, 0],
                 pick_upper_bound=[1, 1, 1],

                 place_lower_bound=[-1, -1, 0],
                 place_upper_bound=[1, 1, 1],

                 ready_pos = [[1, 1, 0.6], [1, 1, 0.6]],
                 
                 fix_pick_height=True,
                 fix_place_height=True,
                 action_dim=2,

                 **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.env = WorldPositionWithVelocityAndGraspingControlWrapper(env, **kwargs)
        self.camera_height = self.env.camera_height

        #### Define the action space
        self.action_dim = action_dim
        self.num_picker = self.env.get_num_picker()
        space_low = np.concatenate([pick_lower_bound, place_lower_bound]*action_dim)\
            .reshape(action_dim, -1)
        space_high = np.concatenate([pick_upper_bound, place_upper_bound]*action_dim)\
            .reshape(action_dim, -1)
        self.action_space = Box(space_low, space_high, dtype=np.float64)

        self.ready_pos = np.asarray(ready_pos)

        ### Each parameters has its class variable
        self.velocity = velocity

        self.motion_trajectory = motion_trajectory
        self.pick_height = pick_height
        self.place_height = place_height

        self.action_step = 0
        self.action_horizon = action_horizon
        self.fix_pick_height = fix_pick_height
        self.fix_place_height = fix_place_height
        self.kwargs = kwargs
        self.action_mode = 'world-pick-and-place'
        self.horizon = self.action_horizon
        self.logger_name = 'standard_logger'
    
    def get_no_op(self):
        return self.ready_pos
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def _process_info(self, info):
        info['no_op'] = self.ready_pos
        info['action_space'] = self.action_space
        info['arena'] = self
        return info

    def reset(self, episode_config=None):
        self.action_step = 0
        info = self.env.reset(episode_config)
        info =  self._process_info(info)
        return info
    
    def get_step(self):
        return self.action_step
    

    def process(self, action):
        action = action.reshape(-1, 6)
        if action.shape[0] == 1:
            self.single_operator = True
            new_action = np.zeros((self.num_picker, 6))
            new_action[0] = action[0]
            new_action[1] = self.ready_pos.reshape(-1, 6)[-1]
            action = new_action

        #print('process action shape', action.shape)

        return {'pick_positions': action.reshape(self.num_picker, 2, -1)[:, 0, :3],
                'place_positions': action.reshape(self.num_picker, 2, -1)[:, 1, :3],
                'action': action}
    
    def step(self, action):
        action_ = self.process(action)
        #print('action', action_)
        pick_positions = action_['pick_positions']
        place_positions = action_['place_positions']
        action = action_['action']
        
        #print('vel', self.velocity)
        velocity_np = np.array([self.velocity]*self.num_picker).reshape(self.num_picker, -1)
        no_cloth_velocity_np = np.array([0.3]*self.num_picker).reshape(self.num_picker, -1)

        
        if self.motion_trajectory in ['rectangular']:
            

            pick_positions_ = pick_positions.copy()
            pick_positions_[:, 2] = self.kwargs['prepare_height']
            #print('prepare height', self.kwargs['prepare_height'])  
            if self.single_operator:
                pick_positions_[1] = self.ready_pos[0, :3]


            actions = [
                ## Go to pick position
                np.concatenate(
                    [
                        pick_positions_, 
                        no_cloth_velocity_np, 
                        np.ones((self.num_picker, 1))
                    ], 
                    axis=1
                ),
                
                ## Lower the picker
                np.concatenate(
                    [
                        pick_positions.copy(),
                        velocity_np, 
                        np.ones((self.num_picker, 1))
                    ], 
                    axis=1
                ),
                
                ## Pick the object
                np.concatenate(
                    [
                        pick_positions.copy(),
                        velocity_np, 
                        -np.ones((self.num_picker, 1))
                    ], 
                    axis=1
                ),

                ## Go and raise to the intermidiate position directly
                np.concatenate(
                    [
                        pick_positions_, 
                        velocity_np, 
                        -np.ones((self.num_picker, 1))
                    ], 
                    axis=1
                ),

  
                ## Go to place position
                np.concatenate(
                    [
                        place_positions, 
                        velocity_np, 
                        -np.ones((self.num_picker, 1))
                    ],     
                    axis=1
                ),
                
                
                ## Place the object
                np.concatenate(
                    [
                        place_positions, 
                        velocity_np, 
                        np.ones((self.num_picker, 1))
                    ],     
                    axis=1
                ),

   
                ## Move back to ready
                
                np.concatenate(
                    [
                        self.ready_pos.copy(), 
                        no_cloth_velocity_np, 
                        np.ones((self.num_picker, 1))
                    ],     
                    axis=1
                ),
            ]
        else:
            raise NotImplementedError
        

        info = self.env.step(actions)
        info = self.env.wait_until_stable()
        
        self.action_step += 1
        done = self.action_step >= self.action_horizon
        #print('action horizon', self.action_horizon)

        info['done'] = done
        return self._process_info(info)
    

    def __getattr__(self, name):
        method = getattr(self.env, name)
        if callable(method):
            # If the attribute is a method, return a bound method
            return method.__get__(self.env, self.env.__class__)
        else:
            # If it's not a method, return the attribute itself
            return method

    def __setattr__(self, name, value):
        if name == 'env':
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)

    def __delattr__(self, name):
        delattr(self.env, name)