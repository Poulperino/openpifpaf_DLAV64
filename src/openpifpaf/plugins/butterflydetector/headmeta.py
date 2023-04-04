"""Head meta objects contain meta information about head networks.

This includes the name, the name of the individual fields, the composition, etc.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Tuple

import openpifpaf

@dataclass
class Butterfly(openpifpaf.headmeta.Base):
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 1
    n_scales: ClassVar[int] = 2

    vector_offsets = [True]
    decoder_min_scale = 0.0
    decoder_seed_mask: List[int] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.keypoints)

@dataclass
class Butterfly_LaplaceWH(openpifpaf.headmeta.Base):
    """Head meta data for a Composite Intensity Field (CIF)."""

    keypoints: List[str]
    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0

    vector_offsets = [True]
    decoder_min_scale = 0.0
    decoder_seed_mask: List[int] = None

    training_weights: List[float] = None

    @property
    def n_fields(self):
        return len(self.keypoints)
