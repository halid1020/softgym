import os
import matplotlib.pyplot as plt
import numpy as np

class TaskWrapper():

    def reset(self, episode_config=None):
        info = self.env.reset(episode_config)
        # we need to deal with this
        info['arena'] = self
        
        return info
    
    def step(self, action):
        info = self.env.step(action)
        info['arena'] = self
        return info
    
    def evaluate(self):
        raise NotImplementedError
    
    def reward(self):
        raise 0
    
    def _save_goal(self):
        eid = self.env.get_episode_id()
        mode = self.env.get_mode()
        if not os.path.exists(self._get_goal_path(eid, mode)):
            os.makedirs(self._get_goal_path(eid, mode))
        plt.imsave(self._get_goal_path(eid, mode) + '/rgb.png', self.goal['rgb'])
        np.save(self._get_goal_path(eid, mode) + '/depth.npy', self.goal['depth'])

    def _get_goal_path(self, eid, mode):
        
        if self.domain == 'mono-square-fabric':
            return '{}/../task_wrappers/rect_fabric/goals/{}/{}'\
                .format(os.environ['SOFTGYM_PATH'], self.task_name, self.domain)
        
        return '{}/../task_wrappers/rect_fabric/goals/{}/{}/initial_{}/{}_eid_{}'\
            .format(os.environ['SOFTGYM_PATH'], self.task_name, self.domain, self.initial, mode, eid)
    
        
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