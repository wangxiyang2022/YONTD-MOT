import argparse, os, tqdm, time, shutil, pickle, sys, torch
from tqdm import tqdm
import numpy as np
from detector.FasterRCNN.predict import FasterRCNNDetector
from detector.pcdet.pcdet.datasets import pcdet_build_dataloader
from detector.CasA.pcdet.models import casa_build_network
from detector.pcdet.pcdet.models import load_data_to_gpu
from detector.CasA.pcdet.datasets import casa_build_dataloader
from detector.CasA.pcdet.config import casa_cfg_from_yaml_file, casa_cfg
from detector.CasA.pcdet.utils import common_utils
from evaluation.waymo.create_submission import testing_set_submission
from mot.tracker import tracker
from mot.tracker.TONTDMOT import YONTDMOT
from mot.tracker.detections import Detections
from mot.utils.utils import  waymo_deconde_dets
from detector.pcdet.pcdet.config import pcdet_cfg_from_yaml_file, pcdet_cfg
from mot.utils.config import Config
from detector.pcdet.pcdet.models import pcdet_build_network
from evaluation.waymo.waymo_eval_track import OpenPCDetWaymoDetectionMetricsEstimator
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="configs/waymo.yaml", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=20, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')
    args = parser.parse_args()
    return args


def track_all_seq(model_3d, model_2d, test_loader):
    start_time = time.time()
    all_results = []
    pose_deuque = deque([0, 0])

    loop = tqdm(test_loader, desc='Processing')

    for i, batch_dict in enumerate(loop):
        frame = batch_dict['sample_idx'][0].astype(np.int64)

        image_path = batch_dict['images_path']
        cam_calibration = batch_dict.pop('cam_calibration')
        cur_pose = batch_dict.pop('pose')
        pose_deuque.pop()
        pose_deuque.appendleft(cur_pose[0])

        load_data_to_gpu(batch_dict)
        data_dict_trk = batch_dict.copy()
        if getattr(args, 'infer_time', False):
            start_time = time.time()

        # ---------------------------------- Perform detection on each frame -------------------------
        with torch.no_grad():
            pred_dicts, ret_dict, batch_dict = model_3d(batch_dict)
            bbox_scores, bbox_lidar, bbox_label, bbox_camera, bbox_image = \
                waymo_deconde_dets(pred_dicts, test_loader.dataset, batch_dict, cfg_mot.class_names)

        # ------------------------------------- Detection initialization ----------------------------
        dets = []
        for d in range(len(bbox_scores)):
            det = Detections(batch_dict['frame_id'],
                             bbox_lidar[d],
                             bbox_scores[d],
                             bbox_label[d],
                             bbox_camera[d],
                             bbox_image[d]
                             )
            dets.append(det)

        # ------------------------------------------- Tracking -------------------------------------
        trackers = tracker.update("waymo_casa",
                                  frame,
                                  dets,
                                  cfg_det,
                                  test_loader.dataset,
                                  data_dict_trk,
                                  model_3d,
                                  model_2d,
                                  cur_pose=cur_pose,
                                  image_path=image_path,
                                  calib=cam_calibration,
                                  last_pose=pose_deuque[-1]
                                  )

        # ----------------------------- Saving the results for evaluation ----------------------------
        trackers.update({"seq_id": np.str_(batch_dict['frame_id'][0][:-4]), "frame_id": frame})
        all_results.append(trackers)

    end_time = time.time()
    print("总耗时%f s" % (end_time-start_time))

    return all_results


if __name__ == "__main__":
    # try:
    #     shutil.rmtree("./output")
    # except OSError as e:
    #     print("Error: %s - %s." % (e.filename, e.strerror))

    args = parse_args()
    cfg_mot = Config(args.cfg_file)

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    log_file_path = os.path.join(cfg_mot.save_path, "logs_file")
    os.makedirs(log_file_path, exist_ok=True)
    #
    logger = common_utils.create_logger()
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    log_file_path = os.path.join(cfg_mot.save_path, "logs_file")
    os.makedirs(log_file_path, exist_ok=True)

    info_path = os.path.join(cfg_mot.DATA_PATH, 'waymo_infos_' + cfg_mot.DATA_SPLIT["test"] + '.pkl')
    infos = pickle.load(open(info_path, 'rb'))

    dist_test = False

    tracker = YONTDMOT(cfg_mot)

    # ------------------------------------------- 3D Detector initialization -------------------------------------
    if cfg_mot.detector_name_3d == "CASA":
        casa_cfg_from_yaml_file(cfg_mot.CASA.cfg, casa_cfg)
        cfg_det = casa_cfg
        test_set, test_loader, sampler = casa_build_dataloader(
            dataset_cfg=cfg_det.DATA_CONFIG,
            class_names=cfg_det.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        model_3d = casa_build_network(model_cfg=cfg_det.MODEL, num_class=len(cfg_det.CLASS_NAMES), dataset=test_set)
        model_3d.load_params_from_file(filename=cfg_mot.CASA.ckpt, logger=logger, to_cpu=False)

    elif cfg_mot.detector_name_3d == "PCDet":
        pcdet_cfg_from_yaml_file(cfg_mot.PCDet[cfg_mot.pcdet_yaml].cfg, pcdet_cfg)
        cfg_det = pcdet_cfg
        test_set, test_loader, sampler = pcdet_build_dataloader(
            dataset_cfg=cfg_det.DATA_CONFIG,
            class_names=cfg_det.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        model_3d = pcdet_build_network(model_cfg=cfg_det.MODEL, num_class=len(cfg_det.CLASS_NAMES), dataset=test_set)

        model_3d.load_params_from_file(
            filename=cfg_mot.PCDet[cfg_mot.pcdet_yaml].ckpt,
            logger=logger,
            to_cpu=dist_test,
            pre_trained_path=args.pretrained_model
           )

    model_3d.cuda()
    model_3d.eval()

    # ------------------------------------------- 2D Detector initialization -------------------------------------
    if cfg_mot["detector_name_2d"] == "FasterRCNN":
        model_2d = FasterRCNNDetector(cfg=cfg_mot)

    # # -------------------------------------------------- Tracking -----------------------------------------------
    all_results = track_all_seq(model_3d, model_2d, test_loader)

    # ---------------------------------------------- Saving the results ------------------------------------------
    with open(os.path.join(cfg_mot.save_path, 'result_dets.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    # ------------------------------------------ Starting Evaluation ---------------------------------------------
    if cfg_mot.DATA_SPLIT["test"] != 'test':
        eval = OpenPCDetWaymoDetectionMetricsEstimator()
        gt_infos_dst = []
        for idx in range(0, len(infos)):
            cur_info = infos[idx]['annos']
            sample_idx = infos[idx]['point_cloud']['sample_idx']
            seq_idx = infos[idx]['point_cloud']['lidar_sequence']
            cur_info['frame_id'] = sample_idx
            cur_info['seq_id'] = seq_idx
            gt_infos_dst.append(cur_info)
        ap_dict = eval.waymo_evaluation(
            all_results, gt_infos_dst, class_name=cfg_mot.class_names, distance_thresh=1000, fake_gt_infos=False
        )
        ap_result_str = '\n'
        for key in ap_dict:
            ap_dict[key] = ap_dict[key][0]
            ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
        logger.info(ap_result_str)

    else:
        testing_set_submission()
