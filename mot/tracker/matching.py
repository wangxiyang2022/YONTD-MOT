# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from mot.tracker.cost_function import iou_3d, siou_2d, giou_3d, dist_3d, siou_3d, iou_2d, dist_iou_3d, giou_2d, diou_2d


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def kitti_cost(detections, trackers, iou_threshold, iou_matrix, cost_chose, reactivete_track=None):
    # dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
    # if len(dets_8corner) > 0:
    #     dets_8corner = np.stack(dets_8corner, axis=0)
    # else:
    #     dets_8corner = []
    # trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in trackers]
    # eucliDistance_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    matched_indices, _ = is_matched_indices(detections, trackers, iou_matrix, iou_threshold, cost_chose, reactivete_track)
    return matched_indices


def nuscenes_cost(detections, trackers, iou_matrix):
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # iou_matrix[d, t] = giou_3d(det, trk, 'giou_3d')
            iou_matrix[d, t] = dist_3d(det, trk)
    matched_indices = greedy_matching(-iou_matrix)
    return matched_indices


def associate_detections_to_trackers(dets, trks, iou_threshold, cost_chose, metric='match_3d', reactivete_track=None):
    # Assigns detections to tracked object (both represented as bounding boxes)
    # detections:  N x 8 x 3
    # trackers:    M x 8 x 3
    # Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    if (len(trks) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), []
    if (len(dets) == 0):
        return np.empty((0, 2), dtype=int), [], np.arange(len(trks))
    iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
    if metric == 'match_3d':
        matched_indices = kitti_cost(dets, trks, iou_threshold, iou_matrix, cost_chose, reactivete_track)
        # matched_indices = nuscenes_cost(detections, trackers, iou_matrix)
    elif metric == 'match_2d':
        dets = np.array([d.to_x1y1x2y2() for d in dets])
        trks = np.array([t.x1y1x2y2() for t in trks])
        matched_indices, _ = is_matched_indices(dets, trks, iou_matrix, iou_threshold, 'siou')

    return is_matched(dets, trks, matched_indices, iou_matrix, iou_threshold)


def trackfusion2Dand3D(tracker_2D, trks_3Dto2D_image, iou_threshold, cost_chose):
    track_indices = list(range(len(tracker_2D)))  # 跟踪对象索引
    detection_indices = list(range(len(trks_3Dto2D_image)))  # 检测对象索引
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    iou_matrix = np.zeros((len(tracker_2D), len(trks_3Dto2D_image)), dtype=np.float32)
    traks_2d = [trk_temp.x1y1x2y2() for trk_temp in tracker_2D]
    matched_indices, _ = is_matched_indices(traks_2d, trks_3Dto2D_image, iou_matrix, iou_threshold, cost_chose)

    return is_matched(tracker_2D, trks_3Dto2D_image, matched_indices, iou_matrix, iou_threshold)


def associate_2D_to_3D_tracking(tracker_2d, tracks_3d, iou_threshold, cost_chose):
    trks_3dto2d_image = [list(i.additional_info[2:6]) for i in tracks_3d]
    matched_track_2d, unmatch_tracker_2d, _ = trackfusion2Dand3D(tracker_2d, trks_3dto2d_image, iou_threshold, cost_chose)
    return matched_track_2d, unmatch_tracker_2d


def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)


def is_matched_indices(dets, trks, iou_matrix, iou_threshold, metric, reactivete_track=None):
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            if metric == 'iou_2d':
                iou_matrix[d, t] = iou_2d(det, trk)             # det: 8 x 3, trk: 8 x 3
            elif metric == 'giou_2d':
                iou_matrix[d, t] = giou_2d(det, trk)
            elif metric == 'siou_2d':
                iou_matrix[d, t] = siou_2d(det, trk)
            elif metric == 'diou_2d':
                iou_matrix[d, t] = diou_2d(det, trk)
            if metric == 'iou_3d':
                iou_matrix[d, t] = iou_3d(det, trk)
            elif metric == 'dist_iou_3d':
                iou_matrix[d, t] = dist_iou_3d(det, trk)
            elif metric == 'siou_3d':
                iou_matrix[d, t] = siou_3d(det, trk, reactivete_track)
            elif metric == 'giou_3d':
                iou_matrix[d, t] = giou_3d(det, trk,  reactivete_track, metric = 'giou_3d')
            elif metric == 'dist_3d':
                iou_matrix[d, t] = dist_3d(det, trk, reactivete_track)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    # matched_indices = greedy_matching(-iou_matrix)
    return matched_indices, iou_matrix


def is_matched(detections, trackers, matched_indices, iou_matrix, iou_threshold):
    matches, unmatched_detections, unmatched_trackers = [], [], []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    # else:
    # 	matches = []
    # 	# calculate Euclidean distance
    # 	for d, det in enumerate(detections):
    # 		for t, trk in enumerate(trackers):
    # 			eucliDistance_matrix[d, t] = eucliDistance(det.bbox[0:3], trk.pose[0:3])
    # 	# eucliDistance_matrix = np.where(eucliDistance_matrix < 1, eucliDistance_matrix, 0)
    #
    # 	if not np.all(eucliDistance_matrix == 0):
    # 		row_ind, col_ind = linear_sum_assignment(eucliDistance_matrix)
    # 		matched_indices = np.stack((row_ind, col_ind), axis=1)
    #
    # 		unmatched_detections = []
    # 		for d, det in enumerate(dets_8corner):
    # 			if d not in matched_indices[:, 0]:
    # 				unmatched_detections.append(d)
    #
    # 		unmatched_trackers = []
    # 		for t, trk in enumerate(trks_8corner):
    # 			if t not in matched_indices[:, 1]:
    # 				unmatched_trackers.append(t)
    #
    # 		for m in matched_indices:
    # 			if eucliDistance_matrix[m[0], m[1]] >= 1.5:
    # 				unmatched_detections.append(m[0])
    # 				unmatched_trackers.append(m[1])
    # 				pass
    # 			else:
    # 				matches.append(m.reshape(1, 2))
    # 		if len(matches) == 0:
    # 			matches = np.empty((0, 2), dtype=int)
    # 		else:
    # 			matches = np.concatenate(matches, axis=0)
    # 	else:
    # 		matches, unmatched_detections, unmatched_trackers = [], [], []
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
