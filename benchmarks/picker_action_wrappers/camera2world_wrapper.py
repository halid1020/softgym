class Camera2WorldWrapper():

    def __init__(self, env):
        
        self.env = env
        
        self.camera_height = env.camera_height
        self.camera_to_world_ratio = self.env.pixel_to_world_ratio
       
    def process(self, action):
        depth = action[:, 2:3].copy() # - 0.02
        action[:, 2] = self.camera_height - action[:, 2]
        action[:, :2] = \
            action[:, :2] * self.camera_to_world_ratio * depth
        return action
     

    def step(self, action):
        action = [self.process(v) for v in action]
        info = self.env.step(action)
        return info
    

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