import numpy as np

from .oracle_towel_folding import OraclTowelFolding
from ..utils import ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD, FOLDING_IoU_THRESHOLD

class AllCornerInwardFolding(OraclTowelFolding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('all-corner-inward-folding')
        if self.folding_noise:
            self.action_types.append('noisy-all-corner-inward-folding')

        self.folding_pick_order = np.asarray([[[0, 0]], [[1, 1]], [[0, 1]], [[1, 0]]])
        self.folding_place_order = np.asarray([[[0.43, 0.43]], [[0.58, 0.58]], 
                                               [[0.43, 0.58]], [[0.58, 0.43]]])
        self.over_ratios = [0, 0, 0, 0]
        self.next_step_threshold = 0.2

    def success(self, info=None):
        if info is None:
            info = self.last_info
        flg = (self.fold_steps != len(self.folding_pick_order))
        flg  = flg and info['largest_particle_distance'] < ALL_CORNER_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            flg = flg and info['canonical_IoU'] >= FOLDING_IoU_THRESHOLD
        return flg