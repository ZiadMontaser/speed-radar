"""
Motion Detection Module - Member A

Implements hybrid motion detection combining:
- Three-frame differencing
- Adaptive background model B_n(x,y)
- Adaptive threshold matrix T_n(x,y)
- Masked subtraction
- Shadow detection and removal

Reference: Adaptive background modeling and masked subtraction with shadow removal
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MotionDetectionConfig:
    """Configuration for motion detection parameters"""
    alpha_background: float = 0.1      # Learning rate for background update
    alpha_threshold: float = 0.05      # Learning rate for threshold update
    initial_threshold: float = 15.0    # Initial T_0(x,y)
    min_threshold: float = 5.0         # Minimum threshold value
    max_threshold: float = 50.0        # Maximum threshold value
    shadow_ratio_min: float = 0.23     # Min ratio I/B for shadow detection
    shadow_ratio_max: float = 0.95     # Max ratio I/B for shadow detection
    morpho_kernel_size: int = 5        # Kernel size for morphological operations
    motion_threshold: int = 3          # Minimum pixel difference for three-frame diff
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MotionDetectionConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class MotionDetector:
    """
    Adaptive motion detection with background subtraction, shadow removal,
    and three-frame differencing.
    """
    
    def __init__(self, config: Optional[MotionDetectionConfig] = None):
        """
        Initialize motion detector.
        
        Args:
            config: MotionDetectionConfig instance or None for defaults
        """
        self.config = config or MotionDetectionConfig()
        self.background = None
        self.threshold_matrix = None
        self.initialized = False
        
    def _initialize_models(self, frame: np.ndarray) -> None:
        """
        Initialize background model and threshold matrix from first frame.
        
        Args:
            frame: Input frame (HxWx3 BGR or HxW grayscale)
        """
        if frame.ndim == 3:
            # Convert BGR to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.astype(np.float32)
        
        self.background = gray.astype(np.float32)
        self.threshold_matrix = np.full_like(
            self.background, 
            self.config.initial_threshold,
            dtype=np.float32
        )
        self.initialized = True
    
    def _compute_three_frame_difference(
        self, 
        frame: np.ndarray, 
        prev_frame: Optional[np.ndarray],
        prev2_frame: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute three-frame differencing for motion detection.
        
        Detects motion by comparing current frame with both previous frames.
        This is robust to sudden lighting changes affecting only one frame.
        
        Args:
            frame: Current frame (HxWx3 BGR or HxW grayscale)
            prev_frame: Previous frame or None
            prev2_frame: Frame two steps back or None
            
        Returns:
            Binary motion mask (HxW uint8)
        """
        if frame.ndim == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            curr_gray = frame.astype(np.float32)
        
        if prev_frame is None or prev2_frame is None:
            # Not enough frames for three-frame differencing; use single difference
            if prev_frame is not None:
                if prev_frame.ndim == 3:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    prev_gray = prev_frame.astype(np.float32)
                diff = np.abs(curr_gray - prev_gray)
            else:
                diff = np.zeros_like(curr_gray)
        else:
            # Three-frame differencing: |I_n - I_n-1| AND |I_n - I_n-2|
            if prev_frame.ndim == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                prev_gray = prev_frame.astype(np.float32)
            
            if prev2_frame.ndim == 3:
                prev2_gray = cv2.cvtColor(prev2_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                prev2_gray = prev2_frame.astype(np.float32)
            
            diff1 = np.abs(curr_gray - prev_gray)
            diff2 = np.abs(curr_gray - prev2_gray)
            # Both differences must indicate motion (AND logic)
            diff = np.minimum(diff1, diff2)
        
        # Threshold motion
        motion_mask = (diff > self.config.motion_threshold).astype(np.uint8) * 255
        return motion_mask
    
    def _background_subtraction(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute background subtraction using adaptive background model.
        
        Args:
            frame: Input frame (HxWx3 BGR or HxW grayscale)
            
        Returns:
            Tuple of (foreground_mask, difference_magnitude)
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        # Ensure background is initialized
        if self.background is None:
            self._initialize_models(frame)
        
        # Compute difference: |I_n(x,y) - B_n(x,y)|
        diff = np.abs(gray - self.background)
        
        # Threshold using adaptive threshold matrix T_n(x,y)
        foreground_mask = (diff > self.threshold_matrix).astype(np.uint8) * 255
        
        return foreground_mask, diff
    
    def _detect_shadows(
        self,
        frame: np.ndarray,
        foreground_mask: np.ndarray
    ) -> np.ndarray:
        """
        Detect and remove shadows using intensity ratio test.
        
        A pixel is a shadow if: 0.23 < I(x,y) / B(x,y) < 0.95
        
        Args:
            frame: Input frame (HxWx3 BGR or HxW grayscale)
            foreground_mask: Current foreground mask
            
        Returns:
            Updated foreground mask with shadows removed
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        if self.background is None:
            return foreground_mask
        
        # Avoid division by zero
        bg_safe = np.maximum(self.background, 1.0)
        
        # Compute intensity ratio: I / B
        intensity_ratio = gray / bg_safe
        
        # Shadow mask: pixels where ratio is in shadow range
        shadow_mask = (
            (intensity_ratio > self.config.shadow_ratio_min) &
            (intensity_ratio < self.config.shadow_ratio_max) &
            (foreground_mask > 0)
        )
        
        # Remove shadows from foreground mask
        shadow_removed_mask = foreground_mask.copy()
        shadow_removed_mask[shadow_mask] = 0
        
        return shadow_removed_mask
    
    def _masked_subtraction(
        self,
        bg_subtraction_mask: np.ndarray,
        three_frame_motion: np.ndarray
    ) -> np.ndarray:
        """
        Combine background subtraction and three-frame differencing.
        
        Uses AND logic to keep only pixels flagged by both methods,
        increasing robustness.
        
        Args:
            bg_subtraction_mask: Mask from background subtraction
            three_frame_motion: Mask from three-frame differencing
            
        Returns:
            Combined foreground mask
        """
        # AND combination: keep only pixels detected by both methods
        combined = cv2.bitwise_and(bg_subtraction_mask, three_frame_motion)
        return combined
    
    def _apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned mask
        """
        kernel_size = self.config.morpho_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # Opening: remove small noise
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing: fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed
    
    def update_background_and_threshold(
        self,
        frame: np.ndarray,
        foreground_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update background model B_n(x,y) and threshold matrix T_n(x,y).
        
        Formulas:
        - B_n(x,y) = (1 - α) * B_{n-1}(x,y) + α * I_n(x,y)  [for background pixels]
        - T_n(x,y) = (1 - α_t) * T_{n-1}(x,y) + α_t * |I_n(x,y) - B_n(x,y)|
        
        Background is only updated for pixels NOT in foreground (background pixels).
        
        Args:
            frame: Input frame (HxWx3 BGR or HxW grayscale)
            foreground_mask: Current foreground mask
            
        Returns:
            Tuple of (updated_background, updated_threshold_matrix)
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        if self.background is None:
            self._initialize_models(frame)
        
        # Background pixels (where foreground_mask == 0)
        background_pixels = foreground_mask == 0
        
        # Update background: B_n(x,y) = (1 - α) * B_{n-1} + α * I_n
        self.background[background_pixels] = (
            (1 - self.config.alpha_background) * self.background[background_pixels] +
            self.config.alpha_background * gray[background_pixels]
        )
        
        # Compute difference for threshold update
        diff = np.abs(gray - self.background)
        
        # Update threshold matrix: T_n = (1 - α_t) * T_{n-1} + α_t * |I_n - B_n|
        self.threshold_matrix = (
            (1 - self.config.alpha_threshold) * self.threshold_matrix +
            self.config.alpha_threshold * diff
        )
        
        # Clamp threshold matrix to valid range
        self.threshold_matrix = np.clip(
            self.threshold_matrix,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        return self.background.copy(), self.threshold_matrix.copy()
    
    def compute_foreground_mask(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        prev2_frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute foreground mask using hybrid approach.
        
        Combines:
        1. Three-frame differencing
        2. Adaptive background subtraction
        3. Shadow removal
        4. Morphological cleaning
        
        Args:
            frame: Current frame (HxWx3 BGR or HxW grayscale)
            prev_frame: Previous frame (optional)
            prev2_frame: Frame two steps back (optional)
            
        Returns:
            Binary foreground mask (HxW uint8, 0 or 255)
        """
        # Initialize on first call
        if not self.initialized:
            self._initialize_models(frame)
        
        # Step 1: Three-frame differencing
        three_frame_motion = self._compute_three_frame_difference(
            frame, prev_frame, prev2_frame
        )
        
        # Step 2: Background subtraction
        bg_subtraction_mask, _ = self._background_subtraction(frame)
        
        # Step 3: Combine both methods
        combined_mask = self._masked_subtraction(bg_subtraction_mask, three_frame_motion)
        
        # Step 4: Shadow detection and removal
        shadow_removed_mask = self._detect_shadows(frame, combined_mask)
        
        # Step 5: Morphological cleanup
        cleaned_mask = self._apply_morphological_operations(shadow_removed_mask)
        
        # Step 6: Update background and threshold for next iteration
        self.update_background_and_threshold(frame, cleaned_mask)
        
        return cleaned_mask
    
    def get_background(self) -> Optional[np.ndarray]:
        """Get current background model (as uint8 grayscale)"""
        if self.background is None:
            return None
        return np.clip(self.background, 0, 255).astype(np.uint8)
    
    def get_threshold(self) -> Optional[np.ndarray]:
        """Get current threshold matrix"""
        if self.threshold_matrix is None:
            return None
        return self.threshold_matrix.copy()
    
    def reset(self) -> None:
        """Reset detector to initial state"""
        self.background = None
        self.threshold_matrix = None
        self.initialized = False