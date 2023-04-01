import glob, os

import cv2
import numpy as np
from skimage import io
from pathlib import Path
from detector.CasA.pcdet.utils import box_utils, calibration_kitti
from detector.CasA.pcdet.datasets.dataset import DatasetTemplate
from dataset_utils.kitti.kitti_oxts import read_image, load_oxts
from detector.CasA.pcdet.utils.box_utils import boxes3d_kitti_camera_to_lidar

class CasaTrackingDataset(DatasetTemplate):
    def __init__(self, seq_id, dataset_cfg, class_names, ob_path=None, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.point_path = Path(os.path.join(self.root_path, "velodyne", str(seq_id).zfill(4)))
        self.image_path = Path(os.path.join(self.root_path, "image_02", str(seq_id).zfill(4)))
        self.calib_path = Path(os.path.join(self.root_path, "calib", str(seq_id).zfill(4))+".txt")
        self.oxts_file = os.path.join(self.root_path, "oxts", str(seq_id).zfill(4) + ".txt")
        self.reorder = [3, 4, 5, 6, 2, 4, 0]
        self.order = [0, 1, 2, 4, 5, 6, 3]
        self.ext = ext
        data_file_list = glob.glob(str(self.point_path / f'*{self.ext}')) if self.point_path.is_dir() else [self.point_path]
        data_file_list.sort()

        image_file_list = glob.glob(str(self.image_path / f'*{".png"}')) if self.image_path.is_dir() else [
            self.image_path]
        image_file_list.sort()

        calib_file_list = glob.glob(str(self.calib_path / f'*{".txt"}')) if self.calib_path.is_dir() else [
            self.calib_path]
        calib_file_list.sort()

        self.sample_file_list = data_file_list
        self.image_file_list = image_file_list
        self.calib_file_list = calib_file_list
        self.ob_path = ob_path

    def get_image_shape(self, index):
        img_file = self.image_file_list[index]
        # assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_calib(self, index):
        calib_file = self.calib_file_list[index]
        # assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    #staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].detach().cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].detach().cpu().numpy()
            pred_labels = box_dict['pred_labels'].detach().cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].detach().cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib) # [x, y, z, l, h, w, r]
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []

        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)

            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        img_shape = self.get_image_shape(index)
        calib = self.get_calib(0)
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': index,
            'calib': calib,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape

        image_path = self.image_file_list[index]
        image = read_image(image_path)
        imu_poses = load_oxts(self.oxts_file)
        if self.ob_path is not None:
            ob_path = os.path.join(self.ob_path, self.seq_name + '.txt')
            if not os.path.exists(ob_path):
                dets_camera = np.zeros(shape=(0, 7))
                dets_lidar = np.zeros(shape=(0, 7))
                dets_score = np.zeros(shape=(0,))
                dets_image = np.zeros(shape=(0, 4))
            else:
                # objects_list = []
                # det_scores = []
                seq_dets_public = np.loadtxt(ob_path, delimiter=',')
                dets_camera = seq_dets_public[seq_dets_public[:, 0] == index, 7:14]
                dets_camera = dets_camera[:, self.reorder]
                dets_lidar = boxes3d_kitti_camera_to_lidar(dets_camera[:, self.order], calib)
                dets_image = seq_dets_public[seq_dets_public[:, 0] == index, 2:6]
                dets_score = np.array([[i]for i in seq_dets_public[seq_dets_public[:, 0] == index, 14]])
                # with open(ob_path) as f:
                #     for each_ob in f.readlines():
                #         infos = re.split(',', each_ob)
                #         # if infos[0] in self.type:
                #         objects_list.append(infos[7:14])
                #         det_scores.append(infos[14])
                # if len(objects_list) != 0:
                #     objects = np.array(objects_list, np.float32)
                #     objects[:, 3:6] = cam_to_velo(objects[:, 3:6], self.V2C)[:, :3]
                #     det_scores = np.array(det_scores, np.float32)
                # else:
                #     objects = np.zeros(shape=(0, 7))
                #     det_scores = np.zeros(shape=(0,))
        else:
            dets_camera = np.zeros(shape=(0, 7))
            dets_score = np.zeros(shape=(0,))
            dets_image = np.zeros(shape=(0, 4))
            dets_lidar = np.zeros(shape=(0, 7))

        track_result_dict = {
            "calib_path": self.calib_path,
            "pts_rect": pts_rect,
            "image": image,
            "image_path": image_path,
            "dets_camera": dets_camera,
            "dets_lidar": dets_lidar,
            "dets_image": dets_image,
            "dets_score": dets_score,
            "imu_poses": imu_poses,
            "bbox_label": "Car",
        }

        # results = {"det":data_dict, "trk":track_result_dict}
        data_dict['track_result_dict'] = track_result_dict
        return data_dict

