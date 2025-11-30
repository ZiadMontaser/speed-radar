from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray  # HxWx3 BGR


@dataclass
class TrackedObject:
    id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    trajectory: List[Tuple[int, Tuple[float, float]]] = field(default_factory=list)
    Fr0: Optional[int] = None
    FrN: Optional[int] = None
    captured_img_ref: Optional[str] = None
    speed_m_s: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Calibration:
    scale_m_per_pixel: Optional[float] = None
    homography: Optional[np.ndarray] = None
    reference_points: Optional[
        List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ] = None
