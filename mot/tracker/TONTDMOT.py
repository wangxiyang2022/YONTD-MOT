# Author: wangxy
# Emial: 1393196999@qq.com
import numpy as np
from mot.tracker.tracker import Tracker


class YONTDMOT(object):
    def __init__(self, cfg):
        '''
        :param max_age:  The maximum frames in which an object disappears.
        :param min_frames: The minimum frames in which an object becomes a trajectory in succession.
        '''
        self.tracker = Tracker(cfg)
        self.cfg = cfg

    def update(self,
               kitti_or_waymo,
               frame,
               dets,
               cfg_det,
               dataset,
               det_data_dict_copy,
               model_3d,
               model_2d,
               imu_poses=None,
               image_path=None,
               calib=None,
               cur_pose=None,
               last_pose=None
               ):
        self.tracker.predict_3d()

        # ========================================= KITTI DATASET  =========================================
        if kitti_or_waymo == "kitti":
            # -------------------------------- ego_motion_compensation ---------------------------
            if (frame > 0) and (calib is not None):
                self.tracker.ego_motion_compensation_kitti(frame, calib, imu_poses)

            # --------------------------------------Track Update----------------------------------
            self.tracker.update(kitti_or_waymo,
                                dets,
                                cfg_det,
                                dataset,
                                det_data_dict_copy,
                                model_3d,
                                model_2d,
                                image_path=image_path,
                                frame=frame,
                                calib=calib
                                )
            outputs = []
            for track in self.tracker.tracks:
                # ---------------Output of Confirmed or self.frame_count <= self.min_hits--------------
                if track.is_confirmed() or frame <= self.cfg[track.bbox_label].min_frames:
                    bbox_2d = track.bbox_image
                    id_3d = track.track_id
                    bbox_label = track.bbox_label
                    bbox_camera = np.array([track.pose_kf[5], track.pose_kf[6], track.pose_kf[4],
                                            track.pose_kf[0], track.pose_kf[1], track.pose_kf[2],
                                            track.pose_kf[3]])   # # [x, y, z, r, l, h, w]
                    outputs.append(np.concatenate(([id_3d], bbox_camera, [bbox_label], bbox_2d, [-1])).reshape(1, -1))
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)

        # ========================================= WAYMO DATASET  =========================================
        elif kitti_or_waymo == "waymo_casa":
            # -------------------------------- ego_motion_compensation ---------------------------------
            if frame > 0:
                self.tracker.ego_motion_compensation_waymo(frame, cur_pose[0], last_pose)

            # --------------------------------------Track Update----------------------------------------
            self.tracker.update(kitti_or_waymo,
                                dets,
                                cfg_det,
                                dataset,
                                det_data_dict_copy,
                                model_3d,
                                model_2d,
                                image_path=image_path,
                                frame=frame,
                                pose_waymo=cur_pose,
                                calib=calib
                                )
            name = []
            score = []
            boxes_lidar = []
            obj_ids = []
            for track in self.tracker.tracks:
                if track.is_confirmed() or frame <= self.cfg[track.bbox_label].min_frames:
                    name.append(track.bbox_label)
                    score.append(track.bbox_score_lidar)
                    boxes_lidar.append(track.bbox_lidar)
                    obj_ids.append(track.track_id)
            # If there is no trajectory in the current frame, a virtual trajectory will be constructed to
            # prevent errors during evaluation. The evaluation code must ensure that there is a trajectory in
            # every frame. May is this a bug in evaluation code ???
            if len(boxes_lidar) == 0:
                name.append(np.array("Vehicle"))
                score.append(np.array(0))
                boxes_lidar.append(np.zeros([1,7])[0])
                obj_ids.append(np.array(9000000000))

            outputs = {'name': np.array(name),
                       "boxes_lidar": np.array(boxes_lidar),
                       "score": np.array(score),
                       "obj_ids": np.array(obj_ids)}

        return outputs



