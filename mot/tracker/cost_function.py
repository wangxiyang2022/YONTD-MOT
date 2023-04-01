# Author: wangxy
# Emial: 1393196999@qq.com

import copy, math
import numpy as np
from numba import jit
from scipy.spatial import ConvexHull
import numpy as np
from numba import njit
PI = np.pi
TWO_PI = 2 * np.pi
from typing import Tuple

def dist_iou_3d(corner1, corner2, reactivate_track=None):
    dist = dist_3d(corner1, corner2)
    iou = iou_3d(corner1, corner2)
    cost = dist + iou*5
    return cost


def giou_3d(box_a, box_b,  reactivate_track=None, metric='giou_3d'):
    ''' Compute 3D/2D bounding box IoU, only working for object parallel to ground

    Input:
        Box3D instances
    Output:
        iou_3d: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    box corner order is like follows
            1 -------- 0 		 top is bottom because y direction is negative
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    rect/ref camera coord:
    right x, down y, front z
    '''
    giou_3 = 0
    if reactivate_track and box_b.is_reactivate():
        for key in reactivate_track:
            if key == box_b.track_id_3d:
                trks = reactivate_track[key]
                if len(np.array(trks).shape) != 1:
                    i = 0
                    for trk in trks:
                        giou_3 += giou_3d(box_a, trk)
                        i += 1
                    giou_3 /= i
                else:
                    giou_3 = giou_3d(box_a, trks)
        return giou_3
    if not isinstance(box_b, list) and not isinstance(box_b, np.ndarray):
        box_b = box_b.pose.tolist()
    else:
        box_b = box_b
    box_a = box_a.bbox.tolist()
    # compute 2D related measures
    boxa_bot, boxb_bot = compute_bottom(box_a, box_b)
    # boxa_bot, boxb_bot = box_a[-5::-1, [0, 2]], box_b[-5::-1, [0, 2]]

    I_2D = compute_inter_2D(boxa_bot, boxb_bot)

    # only needed for GIoU
    if 'giou' in metric:
        C_2D = convex_area(boxa_bot, boxb_bot)

    if '2d' in metric:  # return 2D IoU/GIoU
        U_2D = box_a.w * box_a.l + box_b.w * box_b.l - I_2D
        if metric == 'iou_2d':  return I_2D / U_2D
        if metric == 'giou_2d': return I_2D / U_2D - (C_2D - U_2D) / C_2D
    #[x,y,z,heading,l,w,h]
    elif '3d_det' in metric:  # return 3D IoU/GIoU
        overlap_height = compute_height(box_a, box_b)
        I_3D = I_2D * overlap_height
        U_3D = box_a[5] * box_a[4] * box_a[6] + box_b[5] * box_b[4] * box_b[6] - I_3D
        if metric == 'iou_3d':  return I_3D / U_3D
        if metric == 'giou_3d':
            union_height = compute_height(box_a, box_b, inter=False)
            C_3D = C_2D * union_height
            return I_3D / U_3D - (C_3D - U_3D) / C_3D


def siou_3d(boxA, boxB, reactivate_track=None):
    # giou = iou - (c-aUb)/c
    siou_3 = 0
    if reactivate_track and boxB.is_reactivate():
        for key in reactivate_track:
            if key == boxB.track_id_3d:
                trks = reactivate_track[key]
                if len(np.array(trks).shape) != 1:
                    i = 0
                    for trk in trks:
                        siou_3 += siou_3d(boxA, trk)
                        i += 1
                    siou_3 /= i
                else:
                    siou_3 = siou_3d(boxA, trks)
        return  siou_3
    if not isinstance(boxB, list) and not isinstance(boxB, np.ndarray):
        box_b = boxB.pose.tolist()
    else:
        box_b = boxB
    box_a =  boxA.bbox.tolist()
    # box_a, box_b = boxA.bbox.tolist(), boxB.pose.tolist()
    # compute 2D related measures
    boxa_bot, boxb_bot = compute_bottom(box_a, box_b)
    I_2D = compute_inter_2D(boxa_bot, boxb_bot)  # 这里求得是两个3Dbbox的底部的面的交集所形成的面的面积
    # only needed for GIoU
    C_2D = convex_area(boxa_bot, boxb_bot)  # 这里求得是外接多边形的面积，该多边形是两个3Dbbox的底部的面的外接
    # [x,y,z,heading,l,w,h]
    overlap_height = compute_height(box_a, box_b) # 这里求得是两个3Dbbox的底部的面的交集所形成的面的高
    I_3D = I_2D * overlap_height
    U_3D = box_a[5] * box_a[4] * box_a[6] + box_b[5] * box_b[4] * box_b[6] - I_3D

    volume_a = box_a[5] * box_a[4] * box_a[6]
    volume_b = box_b[5] * box_b[4] * box_b[6]
    volume_ratio = 0 # 两个3D bbox的体积之比
    if volume_a >= volume_b and (volume_a != 0):
        volume_ratio = volume_b / volume_a
    elif volume_b > volume_a and (volume_b != 0):
        volume_ratio = volume_a / volume_b
    distance_bbox_center = distance_two_center(box_a, box_b)
    sigmod_distance_bbox_center = (1 - (1 / (1+np.exp(-distance_bbox_center))))  # 用sigmod函数将距离限制在0-1之间，距离越大越接近于1， 再用1-距离
    union_height = compute_height(box_a, box_b, inter=False)  # 这里求得是外接多边形的面的高
    C_3D = C_2D * union_height   # 外接多边形的体积

    giou_3 = I_3D / U_3D - (C_3D - U_3D) / C_3D
    siou_3 = (I_3D / U_3D) + (volume_ratio + sigmod_distance_bbox_center) * 0.5
    return siou_3


def siou_2d(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 计算box和other的边缘外包框，使得2个box都在框内的最小矩形
    outXmin = min(boxA[0], boxB[0])
    outYmin = min(boxA[1], boxB[1])
    outXmax = max(boxA[2], boxB[2])
    outYmax = max(boxA[3], boxB[3])
    outCenterx = (outXmin + outXmax) / 2
    outCentery = (outYmin + outYmax) / 2
    inCenterxAx = (boxA[0] + boxA[2]) / 2
    inCenterxAy = (boxA[1] + boxA[3]) / 2
    inCenterxBx = (boxB[0] + boxB[2]) / 2
    inCenterxBy = (boxB[1] + boxB[3]) / 2

    boxAArea = (boxA[2] - boxA[0] + 1 ) * (boxA[3] - boxA[1] + 1 )
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    areaRatio = (boxAArea / boxBArea) if boxBArea > boxAArea else (boxBArea / boxAArea)
    if (boxAArea + boxBArea - interArea == 0) or ((outXmax-outXmin)**2 + (outYmax-outYmin)**2 ==0) or (areaRatio == 0):
        return 0
    distanceRatio = math.sqrt((inCenterxBx-inCenterxAx)**2 + (inCenterxBy-inCenterxAy)**2) / \
                    math.sqrt((outXmax-outXmin)**2 + (outYmax-outYmin)**2)
    distanceRatio = 1 - distanceRatio
    aspect_ratioA = (boxA[2]-boxA[0])/(boxA[3]-boxA[1])
    aspect_ratioB = (boxB[2]-boxB[0])/(boxB[3]-boxB[1])
    aspect_ratio = (aspect_ratioA / aspect_ratioB) if aspect_ratioB > aspect_ratioA else (aspect_ratioB / aspect_ratioA)

    siou = (interArea / float(boxAArea + boxBArea - interArea)) + areaRatio * distanceRatio * aspect_ratio

    return siou


def giou_2d(boxA, boxB):
    x1, x2, y1, y2 = boxA[0], boxA[1], boxA[2], boxA[3]  #分别是第一个矩形左右上下的坐标
    x3, x4, y3, y4 = boxB[0], boxB[1], boxB[2], boxB[3]
    iou = iou_2d(boxA, boxB)
    area_C = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    area_1 = (x2-x1)*(y1-y2)
    area_2 = (x4-x3)*(y3-y4)
    sum_area = area_1 + area_2
    w1 = x2 - x1  # 第一个矩形的宽
    w2 = x4 - x3  # 第二个矩形的宽
    h1 = y1 - y2
    h2 = y3 - y4
    W = min(x1, x2, x3, x4) + w1 + w2 - max(x1, x2, x3, x4)  # 交叉部分的宽
    H = min(y1, y2, y3, y4) + h1 + h2 - max(y1, y2, y3, y4)  # 交叉部分的高
    Area = W * H  # 交叉的面积
    add_area = sum_area - Area  # 两矩形并集的面积
    end_area = (area_C - add_area) / area_C  # (c/(AUB))/c的面积
    giou = iou - end_area
    return giou

def diou_2d(boxes1, boxes2):
    '''
        cal DIOU of two boxes or batch boxes
        :param boxes1:[xmin,ymin,xmax,ymax] or
                    [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
        :param boxes2:[xmin,ymin,xmax,ymax]
        :return:
        '''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # cal Intersection
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[..., 0]) + np.square(outer[..., 1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = np.square(boxes1_center[..., 0] - boxes2_center[..., 0]) + \
                 np.square(boxes1_center[..., 1] - boxes2_center[..., 1])

    # cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious


def iou_2d(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if (boxAArea + boxBArea - interArea == 0):
        return 0
    else:
        return interArea / float(boxAArea + boxBArea - interArea)


def iou_3d(corner1, corner2, reactivate_track=None):
    ''' Compute 3D bounding box IoU, only working for object parallel to ground

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    # iou_3 = 0
    # if reactivate_track and corner2.is_reactivate():
    #     for key in reactivate_track:
    #         if key == corner2.track_id_3d:
    #             trks = reactivate_track[key]
    #             if len(np.array(trks).shape) != 1:
    #                 i = 0
    #                 for trk in trks:
    #                     iou_3 += iou_3d(corner1, trk)
    #                     i += 1
    #                 iou_3 /= i
    #             else:
    #                 iou_3 = iou_3d(corner1, trks)
    #     return  iou_3
    if not isinstance(corner2, list) and not isinstance(corner2, np.ndarray):
        box_b = corner2.pose.tolist()
    else:
        box_b = corner2
    box_a =  corner1.bbox.tolist()
    corners1 = convert_3dbox_to_8corner(box_a)
    corners2 = convert_3dbox_to_8corner(box_b)
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    # inter_area = shapely_polygon_intersection(rect1, rect2)
    _, inter_area = convex_hull_intersection(rect1, rect2)

    # volumeDec = dec.bbox[4] * dec.bbox[5] * dec.bbox[6]
    # volumetrk = trk.pose[4] * trk.pose[5] * trk.pose[6]
    # volumeRatio = volumeDec / volumetrk if(volumetrk > volumeDec) else volumetrk / volumeDec
    # try:
    #   _, inter_area = convex_hull_intersection(rect1, rect2)
    # except ValueError:
    #   inter_area = 0
    # except scipy.spatial.qhull.QhullError:
    #   inter_area = 0

    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou_3d = inter_vol / (vol1 + vol2 - inter_vol)
    _, angle_diff = correct_new_angle_and_diff(corner1.bbox[3], corner2.pose[3])
    cos_dist = np.cos(angle_diff)
    # iou_3d = iou_3d * (1 + cos_dist)
    iou_3d = iou_3d * cos_dist
    return iou_3d

def distance_two_center(boxA, boxB):
    corners1 = convert_3dbox_to_8corner(boxA)  # 8 x 3
    corners2 = convert_3dbox_to_8corner(boxB)  # 8 x 3
    c1 = np.average(corners1, axis=0)
    c2 = np.average(corners2, axis=0)

    dist = np.linalg.norm(c1 - c2)
    return dist


def convex_area(boxa_bottom, boxb_bottom):

	# compute the convex area
	all_corners = np.vstack((boxa_bottom, boxb_bottom))
	C = ConvexHull(all_corners)
	convex_corners = all_corners[C.vertices]
	convex_area = PolyArea2D(convex_corners)

	return convex_area


def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def compute_inter_2D(boxa_bottom, boxb_bottom):
	# computer intersection over union of two sets of bottom corner points

	_, I_2D = convex_hull_intersection(boxa_bottom, boxb_bottom)

	# a slower version
	# from shapely.geometry import Polygon
	# reca, recb = Polygon(boxa_bottom), Polygon(boxb_bottom)
	# I_2D = reca.intersection(recb).area

	return I_2D

def compute_bottom(box_a, box_b):
    # obtain ground corners and area, not containing the height
    corners1 = convert_3dbox_to_8corner(box_a)  # 8 x 3
    corners2 = convert_3dbox_to_8corner(box_b)  # 8 x 3

    # get bottom corners and inverse order so that they are in the
    # counter-clockwise order to fulfill polygon_clip
    boxa_bot = corners1[-5::-1, [0, 2]]  # 4 x 2
    boxb_bot = corners2[-5::-1, [0, 2]]  # 4 x 2

    return boxa_bot, boxb_bot


def compute_height(box_a, box_b, inter=True):
    corners1 = convert_3dbox_to_8corner(box_a)  # 8 x 3
    corners2 = convert_3dbox_to_8corner(box_b)  # 8 x 3

    if inter:  # compute overlap height
        ymax = min(corners1[0, 1], corners2[0, 1])
        ymin = max(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)
    else:  # compute union height
        ymax = max(corners1[0, 1], corners2[0, 1])
        ymin = min(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)

    return height

@jit
def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@jit
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s): outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0: return None
    return (outputList)


def eucliDistance(detection, track):
    # coefficient_det = math.sqrt(detection[0] ** 2 + detection[1] ** 2 + detection[2] ** 2)
    # coefficient_trk = math.sqrt(track[0] ** 2 + track[1] ** 2 + track[2] ** 2)
    # x = [i / coefficient_det for i in detection]
    # y = [k / coefficient_trk for k in track]
    # dist = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
    dist = math.sqrt((detection[0] - track[0]) ** 2 + (detection[1] - track[1]) ** 2 + (detection[2] - track[2]) ** 2)
    # dist = 1 /(1+dist)   # Normalization
    return dist


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
        convert it to the 8 corners of the 3D box, the box is in the camera coordinate
        with right x, down y, front z

        Returns:
            corners_3d: (8,3) array in in rect camera coord

        box corner order is like follows
                1 -------- 0         top is bottom because y direction is negative
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7

        rect/ref camera coord:
        right x, down y, front z

        x -> w, z -> l, y -> h
    '''

    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])

    # 3d_det bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d_det bounding box corners  这是什么东西
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d_det bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners, y_corners, z_corners]))  # np.vstack([x_corners,y_corners,z_corners])   3*8按照竖直方向排列
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]  # x
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]  # y
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]  # z

    return np.transpose(corners_3d)


def dist_3d(corner1, corner2, reactivate_track=None) -> float:
    # dist_3 = 0
    # if reactivate_track and corner2.is_reactivate():
    #     for key in reactivate_track:
    #         if key == corner2.track_id_3d:
    #             trks = reactivate_track[key]
    #             if len(np.array(trks).shape) != 1:
    #                 i = 0
    #                 for trk in trks:
    #                     dist_3 += dist_3d(corner1, trk)
    #                     i += 1
    #                 dist_3 /= i
    #             else:
    #                 dist_3 = dist_3d(corner1, trks)
    #     return  dist_3
    # if not isinstance(corner2, list) and not isinstance(corner2, np.ndarray):
    #     coords_1 = np.array(corner2.pose)
    # else:
    #     coords_1 = np.array(corner2)
    # coords_0 =  np.array(corner1.bbox)
    coords_0, coords_1 = np.array(corner1.bbox), np.array(corner2.pose)
    dist = tracking_distance_2d_dims(coords_0, coords_1)
    _, angle_diff = correct_new_angle_and_diff(coords_0[3], coords_1[3])
    assert angle_diff <= np.pi / 2, f"angle_diff {angle_diff}"
    cos_dist = 1 - np.cos(angle_diff)  # in [0, 1] since angle_diff in [0, pi/2]
    return dist * (1 + cos_dist) * (-1)  # multiplier is in [1, 2]



def tracking_distance_2d_dims(coords_0: np.ndarray, coords_1: np.ndarray) -> float:
    return np.linalg.norm(coords_0[np.array((0, 1, 2, 4, 5, 6))] - coords_1[np.array((0, 1, 2, 4, 5, 6))])


def normalize_angle(angle: float) -> float:
    """ Keep the angle in [0; 2 PI] range"""
    while angle < 0:
        angle += TWO_PI
    while angle > TWO_PI:
        angle -= TWO_PI
    assert angle >= 0 and angle <= TWO_PI, f"angle {angle}"
    return angle


def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float) -> Tuple[float, float]:
    """ Return an angle equivalent to the new_angle_to_correct with regards to difference to the current_angle
    Calculate the difference between two angles [-PI/2, PI/2]

    TODO: This can be refactored to just return the difference
    and be compatible with all angle values without worrying about quadrants, but this works for now
    """
    abs_diff = normalize_angle(new_angle_to_correct) - normalize_angle(current_angle)

    if abs(abs_diff) <= PI / 2:  # if in adjacent quadrants
        return new_angle_to_correct, abs_diff

    if abs(abs_diff) >= 3 * PI / 2:  # if in 1st and 4th quadrants and the angle needs to loop around
        abs_diff = TWO_PI - abs(abs_diff)
        if current_angle < new_angle_to_correct:
            return current_angle - abs_diff, abs_diff
        else:
            return current_angle + abs_diff, abs_diff

    # if the difference is > PI/2 and the new angle needs to be flipped
    return correct_new_angle_and_diff(current_angle, PI + new_angle_to_correct)
