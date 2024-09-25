import numpy as np

from .oracle_towel_folding import OraclTowelFolding
from ..utils import *

class RectangularFolding(OraclTowelFolding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('rectangular-folding')
        if self.folding_noise:
            self.action_types.append('noisy-rectangular-folding')
        
        self.sim2real = False if 'sim2real' not in kwargs else kwargs['sim2real']
        print('sim2real folding', self.sim2real)
    
    def init(self, info):
        # self.action_space = info['action_space']
        # self.no_op = info['no_op']
        #print('hello!!!!!')
        H, W = info['cloth_size']
        length = max(H, W)
        width = min(H, W)

        if self.sim2real:

            self.folding_pick_order = [
                [[0, 0]], [[1, 0]],
            ]
            self.folding_place_order = [
                [[0, 0.95]], [[1, 0.95]],
            ]
            self.over_ratios = [0, 0]
            self.next_step_threshold = 0.2
            
        else:
            self.next_step_threshold = 0.08
            phase_steps = ciel(length/width * 2)
            span_per_phase = 1.0/phase_steps
            self.folding_pick_order = []
            self.folding_place_order = []
            self.over_ratios = []
            for i in range(phase_steps):
                target_pos = min(0.9, 1.0*(i+1)*span_per_phase)
                tt_pos = min(1.0, 1.0*(i+1.25)*span_per_phase)
                self.folding_pick_order.extend([[[0, 0]], [[1, 0]], [[0.4, 0]]])
                self.folding_place_order.extend([[[0, target_pos]], [[1, target_pos]], [[0.4, tt_pos]]])
                self.over_ratios.extend([0, 0, 0])
            
            self.folding_pick_order.extend([[[0, 0]], [[1, 0]]])
            self.folding_place_order.extend([[[0, 1]], [[1, 1]]])
            self.over_ratios.extend([0.04, 0.04])

            self.folding_pick_order.extend([[[0, 0]], [[1, 0]]])
            self.folding_place_order.extend([[[0, 1]], [[1, 1]]])
            self.over_ratios.extend([0.04, 0.04])

        self.folding_pick_order = np.asarray(self.folding_pick_order)
        self.folding_place_order = np.asarray(self.folding_place_order)
        #print('shape', self.folding_pick_order)


        if H > W:
            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]
        
        

    def success(self, info=None):
        if info is None:
            info = self.last_info
       
        flg  = info['largest_particle_distance'] < RECTANGLUAR_FOLDING_SUCCESS_THRESHOLD 
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg