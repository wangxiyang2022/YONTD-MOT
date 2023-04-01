# Author: wangxy
# Emial: 1393196999@qq.com
import copy

import numpy as np


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class Calibration(object):
    ''' Calibration matrices and utils
        3d_det XYZ in <label>measure are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        with open(calib_filepath) as f:
            self.P0 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P1 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Projection matrix from rectified camera coord to image2/3 coord
            self.P2 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))
            self.P3 = np.fromstring(f.readline().split(":")[1], sep=" ").reshape((3, 4))

            # Rotation from reference camera coord to rectified camera coord
            line = f.readline()
            self.R0_rect = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 3))

            # Rigid transform from lidar coord to reference camera coord
            line = f.readline()
            self.Tr_lidar_to_cam = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))  # lidar_to_cam
            self.Tr_cam_to_lidar = inverse_rigid_trans(self.Tr_lidar_to_cam)                       # cam_to_lidar

            line = f.readline()
            self.Tr_imu_to_lidar = np.fromstring(line[line.index(" "):], sep=" ").reshape((3, 4))  # imu_to_lidar
            self.Tr_lidar_to_imu = inverse_rigid_trans(self.Tr_imu_to_lidar)                       # lidar_to_imu

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def project_lidar_to_ref(self, pts_3d_lidar):
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)  # nx4
        return np.dot(pts_3d_lidar, np.transpose(self.Tr_lidar_to_cam))

    def project_rect_to_lidar(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in lidar coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_lidar(pts_3d_ref)

    # ===========================
    # ------- 3d_det to 3d_det ----------
    # ===========================
    def imu_to_rect(self, pts_imu):
        pts_velo = self.project_imu_to_lidar(pts_imu)
        pts_ref = self.project_lidar_to_ref(pts_velo)
        pts_rect = self.project_ref_to_rect(pts_ref)
        return pts_rect

    def project_lidar_to_imu(self, pts_3d_lidar):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in IMU coord.
        '''
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)  # nx4
        return np.dot(pts_3d_lidar, np.transpose(self.Tr_lidar_to_imu))

    def rect_to_imu(self, pts_rect):
        pts_velo = self.project_rect_to_lidar(pts_rect)
        pts_imu  = self.project_lidar_to_imu(pts_velo)
        return pts_imu

    def project_imu_to_lidar(self, pts_3d_imu):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in IMU coord.
        '''
        pts_3d_imu = self.cart2hom(pts_3d_imu)  # nx4
        return np.dot(pts_3d_imu, np.transpose(self.Tr_imu_to_lidar))

    def project_ref_to_lidar(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.Tr_cam_to_lidar))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0_rect), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        '''
        Input and Output are nx3 points
        '''
        # pts_3d_ref = Tr_lidar_to_cam * [x y z 1]
        return np.transpose(np.dot(self.R0_rect, np.transpose(pts_3d_ref)))  # R0_rect_rect * Tr_lidar_to_cam * A

    def project_lidar_to_rect(self, pts_3d_lidar):
        pts_3d_ref = self.project_lidar_to_ref(pts_3d_lidar)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d_det to 2d ----------
    # ===========================

    def project_rect_to_image(self, pts_3d_rect):
        '''
            Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_3d_to_image(self, pts_3d_rect):
        '''
            Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))  # nx3     P_rect_2 * R0_rect_rect *Tr_lidar_to_cam * A
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:,0:2]

    def project_lidar_to_image(self, pts_3d_lidar):
        ''' Input: nx3 points in lidar coord.
            Output: nx3 points in image2 coord.
        '''
        pts_3d_rect = self.project_lidar_to_rect(pts_3d_lidar)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d_det ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_lidar(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_lidar(pts_3d_rect)

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object's 3D box with the representation of [h,w,l, x,y,z,theta] and
        convert it to the 8 corners of the 3D box

        Returns:
            corners_3d: (8,3) array in in rect camera coord
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[6])

    # 3d_det bounding box dimensions
    l = bbox3d[2]
    w = bbox3d[1]
    h = bbox3d[0]

    # 3d_det bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d_det bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners, y_corners, z_corners]))  # np.vstack([x_corners,y_corners,z_corners])
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[3]  # x
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[4]  # y
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[5]  # z

    a = np.transpose(corners_3d)
    return np.transpose(corners_3d)


def convert_x1y1x2y2_to_xywh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, w, h]).tolist()


def convert_x1y1x2y2_to_tlwh(bbox):
    '''
    :param bbox: x1 y1 x2 y2
    :return: tlwh: top_left x   top_left y    width   height
    '''
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array(([bbox[0], bbox[1], w, h]))