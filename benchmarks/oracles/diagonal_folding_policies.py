import numpy as np

from .oracle_towel_folding import OraclTowelFolding
from ..utils import *

class DiagonalFolding(OraclTowelFolding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('diagonal-folding')
        if self.folding_noise:
            self.action_types.append('noisy-diagonal-folding')

        self.folding_pick_order = np.asarray([[[0, 0]]]) # step*num_picker*2
        self.folding_place_order = np.asarray([[[0.95, 0.95]]])
        self.over_ratios = [0.2]
    

    def success(self, info=None):
        if info is None:
            info = self.last_info
        flg  = info['largest_particle_distance'] < DIAGONAL_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg