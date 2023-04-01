import numpy as np
import torch
from detector.CasA.pcdet.models.model_utils import model_nms_utils
from mot.tracker.kalman_fileter_3d import KalmanBoxTracker
from mot.tracker.trajectory import Trajectory
from detector.CasA.pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar, boxes3d_kitti_camera_to_imageboxes
from mot.utils.utils import kitti_deconde_dets, waymo_deconde_dets, kitti_2d_confidence_regression, \
    waymo_2d_confidence_regression
from detector.CasA.pcdet.utils import box_utils


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.confidence_his_max = cfg.confidence_his_max
        self.tracks = []
        self.order_back = [0, 1, 2, 4, 5, 6, 3]  # (x, y, z, l, h, w, thet) to (x, y, z, thet, l, h, w)
        self.reorder_back = [0, 1, 2, 4, 5, 6, 3] # (x, y, z, thet, l, h, w) to (x, y, z, l, h, w, thet)
        self.track_id = 0   # The id of 3D track is represented by an even number.
        self.class_names = cfg.class_names

    def predict_3d(self):
        for track in self.tracks:
            track.predict_3d(track.kf_3d)

    def ego_motion_compensation_kitti(self, frame, calib, oxts):
        for track in self.tracks:
            track.ego_motion_compensation_3d_kitti(frame, calib, oxts)

    def ego_motion_compensation_waymo(self, frame, cur_pose, last_pose):
        for track in self.tracks:
            track.ego_motion_compensation_3d_waymo(frame, cur_pose, last_pose)

    def update(self,
               kitti_or_waymo,
               dets,
               cfg_det,
               dataset,
               det_data_dict_copy,
               model_3d,
               model_2d,
               frame=0,
               image_path=None,
               pose_waymo=None,
               calib=None
               ):
        if len(self.tracks):
            # ============================================ KITTI DATASET =============================================
            if kitti_or_waymo == "kitti":
                bbox_camera = self.get_pos("bbox_camera", self.class_names)[:, self.reorder_back]  # [x,y,z,l,h,w,r]
                bbox_lidar = boxes3d_kitti_camera_to_lidar(bbox_camera, calib)   # [x,y,z,l,w,h,r]
                bbox_lidar = torch.from_numpy(bbox_lidar).cuda(non_blocking=True).float()[None]
                with torch.no_grad():
                    pred_dicts, ret_dict, batch_dict = model_3d.forward(det_data_dict_copy, bbox_lidar)
                # ------------------------------------ 2D Trajectory Regression -------------------------------
                scores_2d = kitti_2d_confidence_regression(det_data_dict_copy,
                                                           pred_dicts,
                                                           calib,
                                                           image_path,
                                                           model_2d,
                                                           flag=True
                                                           )

                # ---------------------------------------- Confidence Fusion ----------------------------------
                self.trajectory_confidence_fusion(kitti_or_waymo,
                                                  dataset,
                                                  pred_dicts,
                                                  self.class_names,
                                                  batch_dict,
                                                  scores_2d
                                                  )

            elif kitti_or_waymo == "waymo_casa":
                bbox_lidar = self.get_pos("bbox_camera")[:, self.reorder_back]
                bbox_lidar = torch.from_numpy(bbox_lidar).cuda(non_blocking=True).float()[None]
                with torch.no_grad():
                    pred_dicts, ret_dict, batch_dict = model_3d.forward(det_data_dict_copy, bbox_lidar)

                # ------------------------------------ 2D Trajectory Regression -------------------------------
                scores_2d = waymo_2d_confidence_regression(self.tracks,
                                                           pred_dicts,
                                                           pose_waymo,
                                                           calib,
                                                           image_path,
                                                           model_2d,
                                                           flag=False
                                                           )

                # ---------------------------------------- Confidence Fusion ----------------------------------
                self.trajectory_confidence_fusion(kitti_or_waymo,
                                                  dataset,
                                                  pred_dicts,
                                                  self.class_names,
                                                  batch_dict,
                                                  scores_2d
                                                  )

        # ---------------------- Trajectory Regression Confidence-based Non-Maximum Suppression --------------------
        if len(dets):
            dets = self.trajectory_regression_confidence_nms(dets, cfg_det)
        else:
            for i in range(len(self.tracks)):
                self.tracks[i].mark_missed()

        # ----------------------------------------------- Trajectory Initialization ---------------------------------
        if len(dets):
            for det in dets:
                self.initiate_track_3d(det.bbox_camera,
                                       det.bbox_lidar,
                                       det.bbox_score_lidar,
                                       det.bbox_image,
                                       det.bbox_label,
                                       frame
                                       )

        # -------------------------------------------- Trajectory Post-Processing -----------------------------------
        for t in self.tracks:
            if t.bbox_score_camera > self.cfg[t.bbox_label].confidence_threshold:
                t.state_camera_update()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def initiate_track_3d(self, bbox_camera, bbox_lidar, bbox_score, box_image, bbox_label, frame):
        self.kf_3d = KalmanBoxTracker(bbox_camera)
        pose_kf = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks.append(Trajectory(pose_kf,
                                      bbox_camera,
                                      bbox_lidar,
                                      bbox_score,
                                      box_image,
                                      bbox_label,
                                      frame,
                                      self.kf_3d,
                                      self.track_id,
                                      self.cfg
                                      )
                           )
        self.track_id += 1

    def get_pos(self, flag="bbox_lidar", is_label=["Car", "Vehicle", "Pedestrian", "Cyclist"]):
        """Get the positions of all active tracks."""
        if flag == "bbox_camera":
            if len(self.tracks) != 0:
                bbox = np.array([t.pose_kf for t in self.tracks if t.bbox_label in is_label])
            else:
                bbox = None
        elif flag == "bbox_lidar":
            if len(self.tracks) != 0:
                bbox = np.array([t.bbox_lidar for t in self.tracks if t.bbox_label in is_label])
            else:
                bbox = None
        elif flag == "bbox_image":
            if len(self.tracks) != 0:
                bbox = np.array([t.bbox_image for t in self.tracks if t.bbox_label in is_label])
            else:
                bbox = None
        return bbox

    def trajectory_confidence_fusion(self,
                                     kitti_or_waymo,
                                     dataset,
                                     pred_dicts,
                                     class_names,
                                     batch_dict,
                                     bbox_scores_camera):
        if kitti_or_waymo == "kitti":
            bbox_scores_lidar, bbox_lidar, bbox_label, bbox_camera, bbox_image = \
                kitti_deconde_dets(pred_dicts, dataset, batch_dict, class_names)
        elif kitti_or_waymo == "waymo_casa":
            bbox_scores_lidar, bbox_lidar, bbox_label, bbox_camera, bbox_image = \
                waymo_deconde_dets(pred_dicts, dataset, batch_dict, class_names)
        scores_fusion = []
        for i in range(len(bbox_camera)):
            self.tracks[i].bbox_lidar = bbox_lidar[i]
            self.tracks[i].bbox_image = bbox_image[i]
            self.tracks[i].bbox_score_camera = bbox_scores_camera[i]
            self.tracks[i].bbox_score_lidar = bbox_scores_lidar[i]
            self.tracks[i].score_update(bbox_scores_lidar[i], bbox_scores_camera[i], self.confidence_his_max)
            scores_fusion.append(self.tracks[i].bbox_score_fusion)

        scores_index = np.array(scores_fusion).argsort()[::-1]
        self.tracks = list(np.array(self.tracks)[scores_index])

    def trajectory_regression_confidence_nms(self, dets, cfg_det):
        if len(self.tracks) == 0:
            return dets
        temp = []
        for per_class in self.class_names:
            dets_per_class = [d for d in dets if d.bbox_label == per_class]
            dets_lidar_per_class = torch.from_numpy(np.array([d.bbox_lidar for d in dets_per_class])).cuda().float()
            dets_score_per_class = torch.from_numpy(np.array([d.bbox_score_lidar for d in dets_per_class])).cuda().float()
            if len(dets_per_class):
                trks_per_class = [t for t in self.tracks if t.bbox_label == per_class]
                trks_lidar_per_class = torch.from_numpy(np.array([t.bbox_lidar for t in trks_per_class])).cuda(
                    non_blocking=True).float()
                for i, trk in enumerate(trks_lidar_per_class):
                    nms_track_pos = torch.cat([trk[:][None], dets_lidar_per_class])
                    nms_track_scores = torch.cat(
                        [torch.tensor([1.0]).to(dets_score_per_class.device)[:], dets_score_per_class])
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(nms_track_scores,
                                                                                   nms_track_pos,
                                                                                   cfg_det.MODEL.POST_PROCESSING.NMS_CONFIG)
                    selected = selected[torch.ge(selected, 1)] - 1
                    if selected.shape[0] != len(dets_per_class):
                        order = torch.arange(0, len(dets_per_class)).cuda()
                        superset = torch.cat([order, selected])
                        uniset, count = superset.unique(return_counts=True)
                        mask = (count == 1)
                        result = uniset.masked_select(mask).detach().cpu()

                        trks_per_class[i].update_3d(dets_per_class[result[0]].bbox_camera)
                        trks_per_class[i].bbox_image = dets_per_class[result[0]].bbox_image
                        trks_per_class[i].bbox_lidar = dets_per_class[result[0]].bbox_lidar
                        trks_per_class[i].bbox_score_lidar = dets_per_class[result[0]].bbox_score_lidar

                        dets_score_per_class = dets_score_per_class[selected]
                        dets_per_class = list(np.array(dets_per_class)[selected.detach().cpu().numpy()])
                        dets_lidar_per_class = dets_lidar_per_class[selected]
                        continue
                    dets_score_per_class = dets_score_per_class[selected]
                    dets_per_class = list(np.array(dets_per_class)[selected.detach().cpu().numpy()])
                    dets_lidar_per_class = dets_lidar_per_class[selected]
                    trks_per_class[i].mark_missed()
                    if selected.nelement() == 0:
                        break
                temp += dets_per_class
        return temp
