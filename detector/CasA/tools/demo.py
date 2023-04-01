import argparse
import glob
from pathlib import Path
from detector.CasA.pcdet.utils import calibration_kitti
import numpy as np
import torch

from detector.CasA.pcdet.config import cfg, cfg_from_yaml_file
from detector.CasA.pcdet.datasets import DatasetTemplate
from detector.CasA.pcdet import build_network, load_data_to_gpu
from detector.CasA.pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def get_image_shape(self, index):
        img_file = self.root_path / 'image_2' / ('%s.png' % index)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_calib(self, index):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

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
        calib = self.get_calib(index)
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

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="OpenPCDet/tools/pointrcnn_7870.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


class Detector(object):
    def __init__(self, demo_dataset, logger, args, cfg_det):
        self.args, self.cfg = args, cfg_det
        self.logger = logger
        self.demo_dataset = demo_dataset
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=self.args.ckpt, logger=self.logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def detect(self, frame):
        with torch.no_grad():
            # for idx, data_dict in enumerate(self.demo_dataset):
            data_dict = self.demo_dataset[frame]
            self.logger.info(f'Visualized sample index: \t{frame + 1}')
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)

            pred_boxes = pred_dicts[0]["pred_boxes"]
            pred_scores = pred_dicts[0]["pred_scores"]
            pred_labels = pred_dicts[0]["pred_labels"]

            pred_labels_car = torch.eq(pred_labels, 1)
            pred_boxes_car = pred_boxes[pred_labels_car]
            pred_scores_car = pred_scores[pred_labels_car]

            return pred_boxes_car.detach().cpu().numpy(), pred_scores_car.detach().cpu().numpy()

# if __name__ == '__main__':
#     main()
