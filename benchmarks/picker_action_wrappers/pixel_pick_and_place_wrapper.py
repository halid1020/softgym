import numpy as np
from gym.spaces import Box


from .world_pick_and_place_wrapper \
    import WorldPickAndPlaceWrapper

class PixelPickAndPlaceWrapper():

    def __init__(self, 
                 env,
                 config):

        self.env = WorldPickAndPlaceWrapper(env, config) 
        self.camera_height = self.env.camera_height

        #### Define the action space
        #self.num_picker = self.env.get_num_picker()
        self.action_dim = 2
        self.num_picker = self.env.get_num_picker()
        space_low = np.concatenate([config.pick_lower_bound, config.place_lower_bound]*self.action_dim)\
            .reshape(self.action_dim, -1).astype(np.float32)
        space_high = np.concatenate([config.pick_upper_bound, config.place_upper_bound]*self.action_dim)\
            .reshape(self.action_dim, -1).astype(np.float32)
        self.action_space = Box(space_low, space_high, dtype=np.float32)

        self.no_op = np.ones(self.action_space.shape)
        if self.no_op.shape[0] == 2:
            self.no_op = self.no_op.reshape(2, 2, -1)
            self.no_op[1, :, 0] *= -1
            if self.no_op.shape[2] == 3:
                self.no_op[:, :, 2] = 0.2
            self.no_op = self.no_op.reshape(*self.action_space.shape)
        
        self.ready_pos = np.asarray([[1.5, 1.5, 0.6], [1.5, 1.5, 0.6]])
        ### Each parameters has its class variable
        self.velocity = 0.1

        self.motion_trajectory = 'rectangular'
        self.pick_height = config.pick_height
        self.place_height = config.place_height

        self.action_step = 0
        self.action_horizon = config.action_horizon
        self.fix_pick_height = config.fix_pick_height
        self.fix_place_height = config.fix_place_height
        self.action_mode = 'pixel-pick-and-place'
        self.horizon = self.action_horizon
        self.logger_name = 'pick_and_place_fabric_single_task_logger'
        self.single_operator = False

        self.camera_height = env.camera_height
        self.camera_to_world_ratio = self.env.pixel_to_world_ratio

    
    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    
    def reset(self, episode_config=None):
        info = self.env.reset(episode_config)
        return info
    

    def process(self, action):
        action = action.reshape(-1, 2, 2)
        N = action.shape[0]
        action = action * self.camera_to_world_ratio * self.camera_height

        process_action = np.zeros((N, 2, 3))
        process_action[:, :, :2] = action
        process_action[:, 0, 2] = self.pick_height
        process_action[:, 1, 2] = self.place_height

        return process_action.reshape(-1, 6)

      
    ## It accpet action has shape (num_picker, 2, 3), where num_picker can be 1 or 2
    def step(self, action):
        action_ = self.process(action)
        return self.env.step(action_)
    

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