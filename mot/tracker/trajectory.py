import copy
import numpy as np
from dataset_utils.kitti.kitti_oxts import get_ego_traj, egomotion_compensation_ID


'''
  3D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated with any detections for several frames, it 
  is then regarded as a reappeared trajectory.
'''
class TrackState(object):
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class Trajectory(object):
    def __init__(self, pose_kf, bbox_camera, bbox_lidar, bbox_score_lidar, bbox_image, bbox_label, frame, kf_3d, track_id, cfg, bbox_score_camera=0):
        self.pose_kf = pose_kf
        self.bbox_camera = bbox_camera   #  [x, y, z, r, l, h, w]
        self.bbox_lidar = bbox_lidar
        self.bbox_image = bbox_image
        self.bbox_label = bbox_label
        self.frame_id = frame
        self.kf_3d = kf_3d
        self.track_id = track_id
        self.hits = 1  # 轨迹更新了，年龄就加1， 也就是轨迹被匹配上的次数
        self.state = TrackState.Tentative
        self.n_init = cfg[bbox_label].min_frames       # 连续跟踪多少帧才算一条真正的轨迹
        self._max_age = cfg.max_ages  # 一条轨迹保留的最大帧数
        self.time_since_update = 0     # 轨迹更新表示符
        self.bbox_score_lidar = bbox_score_lidar
        self.bbox_score_lidar_his = []
        self.bbox_score_camera = bbox_score_camera
        self.bbox_score_camera_his = []
        self.bbox_score_fusion = 0

    def score_update(self, bbox_score_lidar, bbox_score_camera, confidence_his_max):
        self.bbox_score_lidar_his.append(bbox_score_lidar)
        self.bbox_score_camera_his.append(bbox_score_camera)
        if len(self.bbox_score_lidar_his) > confidence_his_max:
            del(self.bbox_score_lidar_his[0])
        if len(self.bbox_score_camera_his) > confidence_his_max:
            del(self.bbox_score_camera_his[0])
        self.bbox_score_fusion = np.mean(np.array(self.bbox_score_lidar_his + self.bbox_score_camera_his))

    def predict_3d(self, trk_3d):
        self.pose_kf = trk_3d.predict()

    def update_3d(self, detection_3d):
        self.kf_3d.update(detection_3d)
        self.pose_kf = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.hits += 1
        # self.age += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def state_camera_update(self):
        self.state = TrackState.Confirmed

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_reactivate(self):
        return self.state == TrackState.Reactivate

    def is_track_id_3d(self):
        track_id = self.track_id
        return track_id

    def ego_motion_compensation_3d_kitti(self, frame, calib, oxts):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

        # assert len(self.trackers) == len(trks), 'error'
        ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(oxts, frame, 1, 1, only_fut=True, inverse=True)
        xyz = np.array([self.pose_kf[0], self.pose_kf[1], self.pose_kf[2]]).reshape((1, -1))
        compensated = egomotion_compensation_ID(xyz, calib, ego_rot_imu, ego_xyz_imu, left, right)
        self.pose_kf[0], self.pose_kf[1], self.pose_kf[2] = compensated[0]
        try:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1))
        except:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

    def ego_motion_compensation_3d_waymo(self, frame, cur_pose, lase_pose):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

        # assert len(self.trackers) == len(trks), 'error'
        frame = 1
        oxts = np.array([lase_pose, cur_pose])
        ego_xyz, ego_rot, left, right = get_ego_traj(oxts, frame, 1, 1, only_fut=True, inverse=True)
        xyz = np.array([self.pose_kf[0], self.pose_kf[1], self.pose_kf[2]]).reshape((1, -1))
        compensated = egomotion_compensation_ID_waymo(xyz, ego_rot, ego_xyz)
        self.pose_kf[0], self.pose_kf[1], self.pose_kf[2] = compensated[0]
        try:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1))
        except:
            self.kf_3d.kf.x[:3] = copy.copy(compensated).reshape((-1, 1))


def egomotion_compensation_ID_waymo(traj_id, ego_rot_imu, ego_xyz_imu):
    # correct rotation
    for frame in range(traj_id.shape[0]):
        traj_id[frame, :] = np.matmul(ego_rot_imu[frame], traj_id[frame, :].reshape((3, 1))).reshape((3,))

    # correct transition
    traj_id += ego_xyz_imu[:traj_id.shape[0], :]  # les_frames x 3

    return traj_id