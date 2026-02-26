import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass
from common import image_processing as ip


@dataclass
class MotionDetectionConfig:
    alpha_background: float = 0.05
    alpha_threshold: float = 0.03
    initial_threshold: float = 30.0
    min_threshold: float = 10.0
    max_threshold: float = 60.0

    shadow_ratio_min: float = 0.18
    shadow_ratio_max: float = 0.95

    motion_threshold: int = 30
    morpho_kernel_size: int = 7

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MotionDetectionConfig":
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in config_dict.items() if k in valid})


class MotionDetector:

    def __init__(self, config: Optional[MotionDetectionConfig] = None):
        self.config = config or MotionDetectionConfig()
        self.background = None
        self.threshold_matrix = None
        self.initialized = False

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return frame.astype(np.float32)

    def _initialize_models(self, gray: np.ndarray) -> None:
        self.background = gray.copy()
        self.threshold_matrix = np.full_like(
            gray, self.config.initial_threshold, dtype=np.float32
        )
        self.initialized = True

    def _three_frame_diff(self, curr, prev, prev2):
        if prev is None or prev2 is None:
            return np.zeros(curr.shape, dtype=np.uint8)

        d1 = np.abs(curr - prev)
        d2 = np.abs(curr - prev2)
        motion = np.minimum(d1, d2)

        return (motion > self.config.motion_threshold).astype(np.uint8) * 255

    def _two_frame_diff(self, curr, prev):
        if prev is None:
            return np.zeros(curr.shape, dtype=np.uint8)

        diff = np.abs(curr - prev)
        return (diff > self.config.motion_threshold).astype(np.uint8) * 255

    def _background_subtraction(self, curr):
        diff = np.abs(curr - self.background)
        return (diff > self.threshold_matrix).astype(np.uint8) * 255

    def _remove_shadows(self, curr, fg):
        bg_safe = np.maximum(self.background, 1.0)
        ratio = curr / bg_safe

        shadow = (
            (ratio > self.config.shadow_ratio_min) &
            (ratio < self.config.shadow_ratio_max) &
            (fg > 0)
        )

        fg = fg.copy()
        fg[shadow] = 0
        return fg

    def _cleanup(self, mask):
        k = self.config.morpho_kernel_size
        # kernel = ip.get_structuring_element(ip.MORPH_ELLIPSE, (k, k))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # mask = ip.morphology_ex(mask, ip.MORPH_CLOSE, kernel, iterations=2)
        # mask = ip.morphology_ex(mask, ip.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def _update_models(self, curr, fg):
        bg_pixels = fg == 0

        self.background[bg_pixels] = (
            (1 - self.config.alpha_background) * self.background[bg_pixels] +
            self.config.alpha_background * curr[bg_pixels]
        )

        diff = np.abs(curr - self.background)
        self.threshold_matrix = (
            (1 - self.config.alpha_threshold) * self.threshold_matrix +
            self.config.alpha_threshold * diff
        )

        self.threshold_matrix = np.clip(
            self.threshold_matrix,
            self.config.min_threshold,
            self.config.max_threshold
        )


    def compute_foreground_mask(self, frame, prev_frame=None, prev2_frame=None):

        curr = self._to_gray(frame)
        prev = self._to_gray(prev_frame) if prev_frame is not None else None
        prev2 = self._to_gray(prev2_frame) if prev2_frame is not None else None

        if not self.initialized:
            self._initialize_models(curr)
            return np.zeros(curr.shape, dtype=np.uint8)

        motion_mask = self._three_frame_diff(curr, prev, prev2)

        bg_fg = self._background_subtraction(curr)

        masked_bg = np.bitwise_and(bg_fg, motion_mask)

        two_frame = self._two_frame_diff(curr, prev)

        support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        motion_support = cv2.dilate(motion_mask, support_kernel, iterations=2)
        two_frame = cv2.bitwise_and(two_frame, motion_support)

        # support_kernel = ip.get_structuring_element(ip.MORPH_ELLIPSE, (11, 11))
        # motion_support = ip.dilate(motion_mask, support_kernel, iterations=2)
        # two_frame = np.bitwise_and(two_frame, motion_support)

        fg = masked_bg.copy()
        fg[two_frame > 0] = 255

        fg = self._remove_shadows(curr, fg)

        fg = self._cleanup(fg)

        self._update_models(curr, fg)

        return fg

    def reset(self):
        self.background = None
        self.threshold_matrix = None
        self.initialized = False
