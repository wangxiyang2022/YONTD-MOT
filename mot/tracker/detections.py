# Author: wangxy
# Emial: 1393196999@qq.com

import numpy as np


class Detections(object):
    def __init__(self, frame, bbox_lidar, bbox_score, bbox_label="Car", bbox_camera=None, bbox_image=None):
        self.bbox_camera = bbox_camera   # [x, y, z, r, l, h, w] camera坐标系
        self.bbox_lidar = bbox_lidar     # [x, y, z, l, h, w, r] lidar坐标系
        self.bbox_image = bbox_image
        self.bbox_score_lidar = bbox_score
        self.frame_id = frame
        self.bbox_label = bbox_label

    def to_x1y1x2y2(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret