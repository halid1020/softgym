import numpy as np
import cv2

class WorldPositionWithVelocityAndGraspingControlWrapper():

    def __init__(self, env):
        self.env = env

    def step(self, actions):
        total_steps = 0
        info = {}
        control_singlas = []
        control_frames = []

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
                control_singlas.append(control_signal)
                info = self.env.step(control_signal, process_info=False)
                control_frames.append(info['observation'])
                curr_pos += delta
                total_steps += 1

        control_frames = {k: np.stack([d[k] for d in control_frames]) for k in control_frames[0].keys()}
        # rgb in control frames has shape S x 720 x 720 x 3. I want to resize it to S x 256 x 256 x 3
        control_frames['rgb'] = np.stack([np.asarray(cv2.resize(f, (256, 256))) for f in control_frames['rgb']])
        control_frames['depth'] = np.stack([np.asarray(cv2.resize(f, (256, 256))) for f in control_frames['depth']])
        if len(control_frames['depth'].shape) == 3:
            control_frames['depth'] = np.expand_dims(control_frames['depth'], axis=-1)
        #control_mask = np.stack([np.asarray(cv2.resize(f, (256, 256))) for f in control_frames['mask']])
    
        
        info = self._process_info(info)
        info['control_signals'] = np.stack(control_singlas).astype(np.float32)
        info['control_frames'] = control_frames
       
        info['total_control_steps'] = total_steps
        
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