"""
Custom Image Processing Module

This module provides custom implementations of common image processing functions
to replace OpenCV (cv2) functions. All functions use NumPy for efficient array operations.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List, Optional, Union


# ============================================
# Geometric Transformations
# ============================================


def perspective_transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to points using homography matrix.
    
    Args:
        points: Points array of shape (N, 1, 2) or (N, 2)
        homography: 3x3 homography matrix
    
    Returns:
        Transformed points with same shape as input
    """
    # Handle different input shapes
    original_shape = points.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        pts = points.reshape(-1, 2)
    else:
        pts = points.reshape(-1, 2)
    
    # Convert to homogeneous coordinates
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_homogeneous = np.hstack([pts, ones])
    
    # Apply transformation
    transformed = pts_homogeneous @ homography.T
    
    # Convert back from homogeneous coordinates
    w = transformed[:, 2:3]
    w = np.where(w == 0, 1, w)  # Avoid division by zero
    transformed_pts = transformed[:, :2] / w
    
    # Restore original shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        return transformed_pts.reshape(-1, 1, 2).astype(np.float32)
    else:
        return transformed_pts.astype(np.float32)


# ============================================
# Filtering and Blurring
# ============================================

def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: Input image
        kernel_size: Kernel size (width, height)
        sigma: Standard deviation
    
    Returns:
        Blurred image
    """
    kw, kh = kernel_size
    
    # Create Gaussian kernel
    ax = np.arange(-kw // 2 + 1., kw // 2 + 1.)
    ay = np.arange(-kh // 2 + 1., kh // 2 + 1.)
    xx, yy = np.meshgrid(ax, ay)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution
    if image.ndim == 2:
        return _convolve2d(image, kernel)
    else:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _convolve2d(image[:, :, c], kernel)
        return result


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution using scipy for performance.
    
    Args:
        image: Input 2D array
        kernel: Convolution kernel
    
    Returns:
        Convolved image
    """
    result = ndimage.convolve(image, kernel, mode='nearest')
    return result.astype(image.dtype)


def add_weighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Calculate weighted sum of two arrays.
    Formula: dst = src1 * alpha + src2 * beta + gamma
    
    Args:
        src1: First input array
        alpha: Weight of first array
        src2: Second input array
        beta: Weight of second array
        gamma: Scalar added to each sum
    
    Returns:
        Weighted sum
    """
    result = src1.astype(np.float32) * alpha + src2.astype(np.float32) * beta + gamma
    return np.clip(result, 0, 255).astype(src1.dtype)


# ============================================
# Morphological Operations
# ============================================

def get_structuring_element(shape: int, ksize: Tuple[int, int]) -> np.ndarray:
    """
    Create a structuring element for morphological operations.
    
    Args:
        shape: 0=RECT, 1=CROSS, 2=ELLIPSE
        ksize: Kernel size (width, height)
    
    Returns:
        Structuring element
    """
    kw, kh = ksize
    
    if shape == 0:  # Rectangle
        return np.ones((kh, kw), dtype=np.uint8)
    elif shape == 1:  # Cross
        kernel = np.zeros((kh, kw), dtype=np.uint8)
        kernel[kh//2, :] = 1
        kernel[:, kw//2] = 1
        return kernel
    elif shape == 2:  # Ellipse
        kernel = np.zeros((kh, kw), dtype=np.uint8)
        cy, cx = kh // 2, kw // 2
        for i in range(kh):
            for j in range(kw):
                if ((i - cy)**2 / (kh/2)**2 + (j - cx)**2 / (kw/2)**2) <= 1:
                    kernel[i, j] = 1
        return kernel
    else:
        return np.ones((kh, kw), dtype=np.uint8)


def erode(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode image using morphological erosion.
    
    Args:
        image: Input binary image
        kernel: Structuring element
        iterations: Number of iterations
    
    Returns:
        Eroded image
    """
    result = image.copy()
    
    for _ in range(iterations):
        result = _morphology_op(result, kernel, 'erode')
    
    return result


def dilate(image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Dilate image using morphological dilation.
    
    Args:
        image: Input binary image
        kernel: Structuring element
        iterations: Number of iterations
    
    Returns:
        Dilated image
    """
    result = image.copy()
    
    for _ in range(iterations):
        result = _morphology_op(result, kernel, 'dilate')
    
    return result


def morphology_ex(image: np.ndarray, op: int, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Perform advanced morphological operations.
    
    Args:
        image: Input binary image
        op: Operation type (2=OPEN, 3=CLOSE, 4=GRADIENT, 5=TOPHAT, 6=BLACKHAT)
        kernel: Structuring element
        iterations: Number of iterations
    
    Returns:
        Result of morphological operation
    """
    if op == 2:  # MORPH_OPEN
        temp = erode(image, kernel, iterations)
        return dilate(temp, kernel, iterations)
    elif op == 3:  # MORPH_CLOSE
        temp = dilate(image, kernel, iterations)
        return erode(temp, kernel, iterations)
    elif op == 4:  # MORPH_GRADIENT
        dilated = dilate(image, kernel, iterations)
        eroded = erode(image, kernel, iterations)
        return dilated - eroded
    elif op == 5:  # MORPH_TOPHAT
        opened = morphology_ex(image, 2, kernel, iterations)
        return image - opened
    elif op == 6:  # MORPH_BLACKHAT
        closed = morphology_ex(image, 3, kernel, iterations)
        return closed - image
    else:
        return image


def _morphology_op(image: np.ndarray, kernel: np.ndarray, op: str) -> np.ndarray:
    """
    Apply basic morphological operation (erode or dilate) using scipy for performance.
    
    Args:
        image: Input binary image
        kernel: Structuring element
        op: 'erode' or 'dilate'
    
    Returns:
        Result image
    """
    if op == 'erode':
        return ndimage.grey_erosion(image, footprint=kernel)
    else:  # dilate
        return ndimage.grey_dilation(image, footprint=kernel)


def connected_components(image: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find connected components in binary image using scipy for performance.
    
    Args:
        image: Binary input image
    
    Returns:
        Tuple of (number of labels, labeled image)
    """
    # Use scipy's optimized implementation
    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    labeled, num_features = ndimage.label(image, structure=structure)
    
    return num_features + 1, labeled


# Constants for compatibility
MORPH_RECT = 0
MORPH_CROSS = 1
MORPH_ELLIPSE = 2
MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_GRADIENT = 4
MORPH_TOPHAT = 5
MORPH_BLACKHAT = 6

