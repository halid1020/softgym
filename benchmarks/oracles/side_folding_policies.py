import numpy as np

from .oracle_towel_folding import OraclTowelFolding
from ..utils import *

class SideFolding(OraclTowelFolding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('side-folding')
        if self.folding_noise:
            self.action_types.append('noisy-side-folding')
        #print('HELLLO!!!!!')
    
    def init(self, info):
        #print('HELLLO')
        H, W = info['cloth_size']
        self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]]
            ]
            
        self.folding_place_order = [
            [[0, 0.45]], [[1, 0.45]], [[0.4, 0.5]], [[0, 0.5]], [[1, 0.5]]
        ]

        self.over_ratios = [0, 0, 0.04, 0.04, 0.04]

       
        # Shorten folding distance if W/(H/2) < 1
        small = min(H, W)
        large = max(H, W)
        #print('value', small/(large/2.0) )
        if small/(large/2.0) < 1.5:
            self.folding_pick_order = [
                [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]], [[0.4, 0]], [[0, 0]], [[1, 0]]
            ]
                
            self.folding_place_order = [
                [[0, 0.3]], [[1, 0.3]], [[0.4, 0.3]], [[0, 0.45]], [[1, 0.45]], [[0.4, 0.5]],  [[0, 0.5]], [[1, 0.5]]
            ]
            self.over_ratios = [0, 0, 0, 0, 0, 0.04, 0.04, 0.04]

        self.folding_pick_order = np.asarray(self.folding_pick_order)
        self.folding_place_order = np.asarray(self.folding_place_order)

        if H > W:
            ## flip the pick and place order
            #print('shape', self.folding_pick_order.shape)

            self.folding_pick_order = self.folding_pick_order[:, :, [1, 0]]
            self.folding_place_order = self.folding_place_order[:, :, [1, 0]]

        self.next_step_threshold = 0.05
        
    
    def success(self, info=None):
        if info is None:
            info = self.last_info
        #logging.debug('[oracle, side folding] largest_particle_distance {}'.format(info['largest_particle_distance']))
        #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))

        #flg = (self.fold_steps >= len(self.folding_pick_order)-1)
        flg  = info['largest_particle_distance'] < SIDE_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= 0.7
        return flg