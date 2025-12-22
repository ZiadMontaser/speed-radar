"""
Segmentation Module (SDCS-aligned)

Iterative horizontal + vertical scan segmentation
based on empty scan lines inside candidate regions.
"""

import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path
from common.data_structures import Region
from common import image_processing as ip
import yaml

# Load default config from project root
_config_path = Path(__file__).parent.parent.parent / "config.yaml"
with open(_config_path, "r") as f:
    _default_config = yaml.safe_load(f)

SEG = _default_config["segmentation"]


# -------------------------
# Utility
# -------------------------
def has_gap(projection, min_gap):
    count = 0
    for v in projection:
        if v == 0:
            count += 1
            if count >= min_gap:
                return True
        else:
            count = 0
    return False


# -------------------------
# Horizontal scan
# -------------------------
def horizontal_scan(mask, bbox):
    x, y, w, h = bbox
    region = mask[y:y+h, x:x+w]

    projection = np.sum(region > 0, axis=0)
    min_gap = SEG["min_run_width"]

    splits = []
    start = None
    gap_count = 0

    for i, v in enumerate(projection):
        if v > 0:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap:
                    splits.append((x + start, y, i - start - gap_count + 1, h))
                    start = None

    if start is not None:
        splits.append((x + start, y, w - start, h))

    return splits if len(splits) > 1 else [bbox]


# -------------------------
# Vertical scan
# -------------------------
def vertical_scan(mask, bbox):
    x, y, w, h = bbox
    region = mask[y:y+h, x:x+w]

    projection = np.sum(region > 0, axis=1)
    min_gap = SEG["min_run_width"]

    splits = []
    start = None
    gap_count = 0

    for i, v in enumerate(projection):
        if v > 0:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= min_gap:
                    splits.append((x, y + start, w, i - start - gap_count + 1))
                    start = None

    if start is not None:
        splits.append((x, y + start, w, h - start))

    return splits if len(splits) > 1 else [bbox]


# Merge vertical scans

def merge_vertical_fragments(regions):
    """
    Merge vertically stacked regions that likely belong to the same vehicle
    """
    merged = []
    used = set()

    for i, r1 in enumerate(regions):
        if i in used:
            continue

        x1, y1, w1, h1 = r1.bbox
        cx1, cy1 = r1.centroid

        group = [r1]
        used.add(i)

        for j, r2 in enumerate(regions):
            if j <= i or j in used:
                continue

            x2, y2, w2, h2 = r2.bbox
            cx2, cy2 = r2.centroid

            # --- Conditions for SAME CAR ---
            horizontally_close = abs(cx1 - cx2) < max(w1, w2) * 0.5
            vertically_close = abs(cy1 - cy2) < max(h1, h2) * 1.2
            similar_width = min(w1, w2) / max(w1, w2) > 0.6

            if horizontally_close and vertically_close and similar_width:
                group.append(r2)
                used.add(j)

        # ---- Merge group into one region
        if len(group) == 1:
            merged.append(group[0])
        else:
            xs = []
            ys = []
            masks = []
            area = 0

            for g in group:
                x, y, w, h = g.bbox
                xs.extend([x, x + w])
                ys.extend([y, y + h])
                area += g.area
                masks.append(g.mask)

            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            cx = np.mean([g.centroid[0] for g in group])
            cy = np.mean([g.centroid[1] for g in group])

            merged.append(
                Region(
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    centroid=(cx, cy),
                    area=area,
                    mask=None  
                )
            )

    return merged


# -------------------------
# Main segmentation
# -------------------------
def segment_foreground(mask: np.ndarray, config: Optional[dict] = None) -> List[Region]:
    # Use provided config or fall back to default
    seg_config = config.get("segmentation", SEG) if config else SEG

    min_area = seg_config["min_area"]
    padding = seg_config.get("padding", 2)
    max_iter = seg_config["max_iterations"]

    # num, labels = ip.connected_components(mask)
    num, labels = cv2.connectedComponents(mask)

    regions = []
    for lbl in range(1, num):
        ys, xs = np.where(labels == lbl)
        if len(xs) == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        regions.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    for _ in range(max_iter):
        new_regions = []
        changed = False

        for bbox in regions:
            h_split = horizontal_scan(mask, bbox)
            for hb in h_split:
                v_split = vertical_scan(mask, hb)
                if len(v_split) > 1:
                    changed = True
                new_regions.extend(v_split)

        if not changed:
            break
        regions = new_regions

    output = []
    for x, y, w, h in regions:
        region_mask = mask[y:y+h, x:x+w]
        area = np.sum(region_mask > 0)

        if area < min_area:
            continue

        ys, xs = np.where(region_mask > 0)
        cx = x + xs.mean()
        cy = y + ys.mean()

        output.append(
            Region(
                bbox=(x - padding, y - padding, w + 2*padding, h + 2*padding),
                centroid=(cx, cy),
                area=int(area),
                mask=region_mask
            )
        )
    output = merge_vertical_fragments(output)
    return output
