import numpy as np

class WorldPositionWithVelocityAndGraspingControlWrapper():

    def __init__(self, env, **kwargs):
        self.env = env
        self.kwargs = kwargs

    def step(self, actions):
        total_steps = 0
        info = {}
        
        for action in actions:
            
            pickers_position = self.env.get_picker_position()
            target_position = action[:, :3].copy()
            target_position[:, [1, 2]] = target_position[:, [2, 1]]
            velocity = action[:, 3]

            delta = target_position - pickers_position
            distance = np.linalg.norm(delta, axis=1)
            num_step = np.ceil(np.max(distance / velocity)).astype(int)

            delta /= num_step
            norm_delta = np.linalg.norm(delta, axis=1, keepdims=True)

            curr_pos = pickers_position.copy()

            for i in range(num_step + 1):
                dist = np.linalg.norm(target_position - curr_pos, axis=1, keepdims=True)
                mask = dist < norm_delta
                delta = np.where(mask, target_position - curr_pos, delta)
                
                control_signal = np.hstack([delta, action[:, 4:5]])
                info = self.env.step(control_signal, process_info=(i == num_step))
                curr_pos += delta
                total_steps += 1
            
            #print(f'action {action} num steps {num_step} took {time.time() - start_time:.6f} seconds')
        
        info['total_control_steps'] = total_steps
        info = self._process_info(info)
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