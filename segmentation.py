"""
Segmentation Module

Multi-iteration horizontal+vertical segmentation for separating 
foreground objects in a binary mask.
"""

import numpy as np
from typing import List

from data_structures import Region
import cv2
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def horizontal_scan(mask: np.ndarray, bbox: tuple) -> List[tuple]:

    x, y, w, h = bbox
    
    region_mask = mask[y:y+h, x:x+w]
    
    mo = config["segmentation"]["morph_open"]
    mc = config["segmentation"]["morph_close"]
    if mo > 0:
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mo,mo)))
    if mc > 0:
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mc,mc)))

    dil = config.get("horizontal_dilate", 1)
    if dil > 1:
        kernel = np.ones((1, dil), np.uint8)
        region_mask = cv2.dilate(region_mask, kernel)

    h_projection = np.sum(region_mask, axis=0)
    
    splits = []
    in_object = False
    start_x = 0
    


    for i, val in enumerate(h_projection):
        if val > 0 and not in_object:
            in_object = True
            start_x = i
        elif val == 0 and in_object:
            in_object = False
            if i - start_x > 0:
                splits.append((x + start_x, y, i - start_x, h))
    
    if in_object:
        splits.append((x + start_x, y, len(h_projection) - start_x, h))
    
    return splits if splits else [bbox]



def vertical_scan(mask: np.ndarray, bbox: tuple) -> List[tuple]:

    x, y, w, h = bbox
    
    region_mask = mask[y:y+h, x:x+w]
    
    mo = config["segmentation"]["morph_open"]
    mc = config["segmentation"]["morph_close"]
    if mo > 0:
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mo,mo)))
    if mc > 0:
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(mc,mc)))

    dil = config.get("vertical_dilate", 1)
    if dil > 1:
        kernel = np.ones((dil, 1), np.uint8)
        region_mask = cv2.dilate(region_mask, kernel)

    v_projection = np.sum(region_mask, axis=1)
    
    # Find gaps (rows with zero sum)
    splits = []
    in_object = False
    start_y = 0
    
    for i, val in enumerate(v_projection):
        if val > 0 and not in_object:
            in_object = True
            start_y = i
        elif val == 0 and in_object:
            in_object = False
            if i - start_y > 0:
                splits.append((x, y + start_y, w, i - start_y))
    
    if in_object:
        splits.append((x, y + start_y, w, len(v_projection) - start_y))
    
    return splits if splits else [bbox]


def segment_foreground(mask: np.ndarray, config: dict) -> List[Region]:
    min_area = config['segmentation']['min_area']
    
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return [] 
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    initial_bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    regions = [initial_bbox]
    
    max_iterations = 50 
    for _ in range(max_iterations):
        new_regions = []
        for bbox in regions:
            h_splits = horizontal_scan(mask, bbox)
            for h_bbox in h_splits:
                v_splits = vertical_scan(mask, h_bbox)
                new_regions.extend(v_splits)
        
        if set(new_regions) == set(regions):
            break
        regions = new_regions
    
    result = []
    for bbox in regions:
        x, y, w, h = bbox
        region_mask = mask[y:y+h, x:x+w]
        area = np.sum(region_mask > 0)
        
        if area >= min_area:
            local_coords = np.where(region_mask > 0)
            if len(local_coords[0]) > 0:
                cy = y + np.mean(local_coords[0])
                cx = x + np.mean(local_coords[1])
                result.append(Region(
                    bbox=bbox,
                    centroid=(cx, cy),
                    area=int(area),
                    mask=region_mask
                ))
    
    return result
