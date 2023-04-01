import numpy as np
import torch
from waymo_open_dataset.utils import box_utils

from detector.pcdet.pcdet.utils import box_utils as box_utils_kitti
from waymo_open_dataset import dataset_pb2
import tensorflow as tf
from waymo_open_dataset.camera.ops import py_camera_model_ops


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [ x, y, z, l, h, w, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    if type(boxes3d) != torch.Tensor:
        boxes3d = torch.from_numpy(boxes3d).cuda(non_blocking=True).float()

    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 3] / 2, boxes3d[:, 5] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def kitti_deconde_dets(pred_dicts, dataset, batch_dict, class_names):
    annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names)
    bbox_xyz = annos[0]["location"]
    bbox_lhw = annos[0]["dimensions"]
    bbox_rotation_y = annos[0]["rotation_y"]
    bbox_image = annos[0]["bbox"]
    bbox_scores = annos[0]["score"]
    lwh_to_lhw = [0,1,2,3,5,4,6]
    bbox_lidar = annos[0]["boxes_lidar"][:,lwh_to_lhw] # [x, y, z, l, w, h, r] to [x, y, z, l, h, w, r]
    bbox_camera = np.hstack((bbox_xyz, bbox_rotation_y[:, None], bbox_lhw))  # [x, y, z, r, l, h, w]
    bbox_label = annos[0]["name"]

    return bbox_scores, bbox_lidar, bbox_label, bbox_camera, bbox_image


def waymo_deconde_dets(pred_dicts, det_dataset, batch_dict, class_names):
    reorder = [0, 1, 2, 6, 3, 4, 5]
    annos = det_dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names)
    bbox_scores = annos[0]["score"]
    bbox_lidar = annos[0]["boxes_lidar"]
    bbox_label = annos[0]["name"]
    bbox_camera = bbox_lidar[:, reorder]
    bbox_image = np.zeros([len(bbox_scores), 4])

    return bbox_scores, bbox_lidar, bbox_label, bbox_camera, bbox_image


def kitti_2d_confidence_regression(data_dict_trk, pred_dicts, calib, image_path, model_2d, flag=True):
    bbox_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()

    if flag:
        # ------------------------- Project the regressed trajectory onto the image--------------------
        image_shape = data_dict_trk['image_shape'][0].detach().cpu().numpy()

        pred_boxes_camera = box_utils_kitti.boxes3d_lidar_to_kitti_camera(bbox_lidar, calib)

        pred_boxes_img = box_utils_kitti.boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )
        pred_boxes_img = torch.from_numpy(pred_boxes_img).cuda(non_blocking=True).float()

        # ------------------------------------ 2D Trajectory Regression -------------------------------
        boxes_2d, scores_2d = model_2d.predict(image_path, pred_boxes_img)

        return scores_2d

    else:
        return np.zeros([len(bbox_lidar)])


def waymo_2d_confidence_regression(cur_trks, pred_dicts, cur_pose, cam_calib, images_path, model_2d, flag=True):
    bbox_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
    if flag:
        corners = box_utils.get_upright_3d_box_corners(bbox_lidar).numpy()  # [N, 8, 3]
        corners = corners.reshape(-1, 3)
        from collections import defaultdict
        box_img_dict = defaultdict(list)
        box_img_dict_id = defaultdict(list)

        no_camera_box_id = []
        for key in cam_calib[0].keys():
            projected_corners = project_vehicle_to_image(cur_pose[0], cam_calib[0][key],
                                                         corners)
            projected_corners = projected_corners.reshape(-1, 8, 3)
            for i in range(len(projected_corners)):
                projected_corners_temp = projected_corners[i]
                u, v, ok = projected_corners_temp.transpose()
                ok = ok.astype(bool)
                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u = u[ok]
                v = v[ok]

                # Clip box to image bounds.
                u = np.clip(u, 0, cam_calib[0][key].width)
                v = np.clip(v, 0, cam_calib[0][key].height)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                lt = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                box_img_dict[key].append([lt[0], lt[1], lt[0] + width, lt[1] + height])
                box_img_dict_id[key].append(i)

            img_path_input = images_path[0][key + '_path']
            bbox_input = torch.from_numpy(np.array(box_img_dict[key])).cuda(non_blocking=True).float()
            bbox_ids = box_img_dict_id[key]
            if bbox_input.shape[0] == 0:
                continue
            no_camera_box_id.extend(bbox_ids)
            with torch.no_grad():
                boxes_2d, scores_2d = model_2d.predict(img_path_input, bbox_input)

            for i in range(len(bbox_ids)):
                cur_trks[bbox_ids[i]].bbox_score_camera = scores_2d[i]

        # Since The FOV of Waymo's camera is only about 250 degrees,
        # there are some objects connot be projected onto the camera
        # For this objects, 3D confidence will be used instead of 2D confidence
        sorted(no_camera_box_id)
        for i in range(len(cur_trks)):
            if i not in no_camera_box_id:
                cur_trks[i].bbox_score_camera = cur_trks[i].bbox_score_lidar
        bbox_scores_camera = np.array([t.bbox_score_camera for t in cur_trks])
        return bbox_scores_camera

    else:
        return np.zeros([len(bbox_lidar)])


def project_vehicle_to_image(pose_matrix, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate
        system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    # Transform points from vehicle to world coordinate system (can be
    # vectorized).
    world_points = np.zeros_like(points)
    ones = np.ones(shape=(points.shape[0], 1))
    point_temp = np.concatenate([points, ones], -1)
    world_points = np.matmul(point_temp, pose_matrix.T)[:, 0:3]

    # for i, point in enumerate(points):
    #     cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    #     world_points[i] = (cx, cy, cz)

    # Populate camera image metadata. Velocity and latency stats are filled with
    # zeroes.
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
        [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant([
        calibration.width,
        calibration.height,
        dataset_pb2.CameraCalibration.GLOBAL_SHUTTER,
    ],
        dtype=tf.int32)
    camera_image_metadata = list(pose_matrix.flatten()) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok).
    return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                              camera_image_metadata,
                                              world_points).numpy()



