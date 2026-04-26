"""
Importing this package registers all representations automatically.
The order of imports determines nothing — keys are always sorted when rendered.
"""

from representations.base import all_representations, get_repr, registered_keys

# Side-effect imports: each module registers its classes into the registry
import representations.ascii_repr
import representations.coordinate_repr
import representations.graph_repr
import representations.tensor_repr
import representations.bitboard_repr

__all__ = ["all_representations", "get_repr", "registered_keys"]