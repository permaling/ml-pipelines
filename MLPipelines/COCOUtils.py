import math
from typing import List, Union, Tuple

import numpy as np
from shapely.geometry import Polygon


def rbb_coco_from_seg(seg: List[float],
                      min_area=4.0, min_hw=1.0, filter_small=False) -> Union[List, List]:
    rbb_corners = minrect_from_segmentation(seg)
    if not rbb_corners:
        return None
    rcoco = corners2rotatedbbox(rbb_corners, min_area, min_hw, filter_small)
    coco = segmentation2bbox(np.array(seg).reshape(-1, 2))
    return rcoco, coco

def minrect_from_segmentation(segmentation):
    points = []
    for i in range(0, len(segmentation), 2):
        points.append([segmentation[i], segmentation[i+1]])
    points.append(points[0])
    poly = Polygon(points)
    min_rect = poly.minimum_rotated_rectangle
    if not hasattr(min_rect, "exterior"):
        return None
    return list(min_rect.exterior.coords)

def calc_bearing(p0, p1):
    np0 = np.array(p0)
    np1 = np.array(p1)
    d = np1 - np0
    theta = math.atan2(d[1], d[0])
    if theta > math.pi:
        theta = theta - 2*math.pi
    if theta < -math.pi:
        theta = theta + 2*math.pi
    return theta

# From https://developer.nvidia.com/blog/detecting-rotated-objects-using-the-odtk/
def _corners2rotatedbbox(corners):
   centre = np.mean(np.array(corners), 0)
   theta = calc_bearing(corners[0], corners[1])
   rotation = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
   out_points = np.matmul(corners - centre, rotation) + centre
   x, y = list(out_points[0,:])
   w, h = list(out_points[2, :] - out_points[0, :])
   return [x, y, w, h, theta]

def corners2rotatedbbox(corners, min_area, min_hw, filter_small):
    for idx in range(len(corners)-1):
        permute_corners = corners[idx:-1] + corners[0:idx]
        xmin, ymin, orig_w, orig_h, theta = \
            _corners2rotatedbbox(permute_corners)
        w = orig_w
        h = orig_h
        if w > 0 and h > 0:
            if w < min_hw or h < min_hw or w*h < min_area:
                if filter_small:
                    continue
                # make sure w and h are reasonable first
                w = max(0.1, w)
                h = max(0.1, h)
                if w*h < min_area:
                    scale = math.sqrt(min_area/w/h)
                    w = w * scale
                    h = h * scale
                w = max(min_hw, w)
                h = max(min_hw, h)
                print("updated w, h {} {} ->  {} {}: min_hw {} min_area {}".
                      format(orig_w, orig_h, w, h, min_hw, min_area))
            return xmin, ymin, w, h, theta

    xmin, ymin, w, h, theta = _corners2rotatedbbox(corners[0:-1])
    if not filter_small:
        w = max(min_hw, w)
        h = max(min_hw, h)
        print("no-corner: updated {} {} ->  {} {}: min_hw {} min_area {}".
              format(orig_w, orig_h, w, h, min_hw, min_area))
        return xmin, ymin, w, h, theta

def segmentation2bbox(seg: np.array) -> Tuple:
    xmin = np.min(seg[:, 0])
    ymin = np.min(seg[:, 1])
    xmax = np.max(seg[:, 0])
    ymax = np.max(seg[:, 1])
    w = xmax - xmin
    h = ymax - ymin
    return xmin, ymin, w, h

def aabb2poly(aabb: np.array) -> Tuple:
    xmin = aabb[0]
    ymin = aabb[1]
    xmax = aabb[0] + aabb[2]
    ymax = aabb[1] + aabb[3]
    return np.array([
        xmin, ymin,
        xmax, ymin, 
        xmax, ymax,
        xmin, ymax
    ])
