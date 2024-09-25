import numpy as np

from .folding_wrapper import FoldingWrapper
from ...constants import *
from ....oracles.corners_egde_inward_folding_policies import CornersEdgeFoldingExpertPolicy

class CornersEdgeInwardFoldingWrapper(FoldingWrapper):

    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', 
                 initial='crumple', 
                 action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.env = env
        self.domain = domain
        self.initial = initial
        self.task_name = 'corners-edge-inward-folding'
        self.oracle_policy = CornersEdgeFoldingExpertPolicy()
        self.action = action
        
    def reset(self, episode_config=None):
        info_ = self.env.reset(episode_config)
        episode_config = self.env.get_episode_config()

        H, W = self.env.get_cloth_size()
        num_particles = H*W
        assert H == W, "Only allow square fabric"

        particle_grid_idx = np.array(list(range(num_particles))).reshape(H, W)#.T  # Reversed index here

        self.fold_groups = []

        if H == W:
            for _ in range(4):
                X = particle_grid_idx.shape[0]
                x_split = X // 4
                group_a_edge = particle_grid_idx[X-x_split:].flatten()
                group_b_edge = np.flip(particle_grid_idx[X-2*x_split:X-x_split], axis=0).flatten()

                x_split = X // 2
                upper_triangle_ids = np.triu_indices(x_split)
                
                group_a_corner = np.concatenate([
                    particle_grid_idx[:x_split, :x_split][upper_triangle_ids].flatten(), 
                    particle_grid_idx[:x_split:, X-x_split:][upper_triangle_ids].flatten()])
                group_b_corner = np.concatenate([
                    np.flip(np.flip(particle_grid_idx[:x_split, :x_split], axis=0), axis=1).T[upper_triangle_ids].flatten(),  
                    particle_grid_idx[:x_split, X-x_split:].T[upper_triangle_ids].flatten()])
                
                group_a = np.concatenate([group_a_edge, group_a_corner])
                group_b = np.concatenate([group_b_edge, group_b_corner])

                self.fold_groups.append((group_a, group_b))
                particle_grid_idx = np.rot90(particle_grid_idx)
       

        self.load_goals(self.env.get_episode_id(), self.env.get_mode())

        info_ = self.env.reset(episode_config)

        return self._process_info(info_)
    
    def success(self):
        is_success = self._largest_particle_distance() < CORNERS_EDGE_INWARD_FOLDING_SUCCESS_THRESHOLD
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= FOLDING_IoU_THRESHOLD
        
        return is_success
    