from .random_policy import RandomPolicy
from .random_pick_and_place_policy import RandomPickAndPlacePolicy

from .real2sim_smoothing import Real2SimSmoothing
from .oracle_towel_smoothing import OracleTowelSmoothing

from .one_corner_inward_folding_policies import OneCornerInwardFolding
from .all_corner_inward_folding_policies import AllCornerInwardFolding
from .double_corner_inward_folding_policies import DoubleCornerInwardFolding
from .side_folding_policies import SideFolding
from .double_side_folding_policies import DoubleSideFolding
from .rectangular_folding_policies import RectangularFolding
from .corners_egde_inward_folding_policies import CornersEdgeInwardFolding
from .diagonal_folding_policies import DiagonalFolding
from .diagonal_cross_folding_policies import DiagonalCrossFolding


NAME2POLICY = {
    "random": RandomPolicy,
    "random_pick_and_place": RandomPickAndPlacePolicy,
    'real2sim-smoothing': Real2SimSmoothing,
    'oracle-towel-smoothing': OracleTowelSmoothing,
    'one-corner-inward-folding': OneCornerInwardFolding,
    'all-corner-inward-folding': AllCornerInwardFolding,
    'double-corner-inward-folding': DoubleCornerInwardFolding,
    'side-folding': SideFolding,
    'double-side-folding': DoubleSideFolding,
    'rectangular-folding': RectangularFolding,
    'corners-edge-inward-folding': CornersEdgeInwardFolding,
    'diagonal-folding': DiagonalFolding,
    'diagonal-cross-folding': DiagonalCrossFolding,
}