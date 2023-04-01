import argparse, os, tqdm, time, shutil, sys
from torch.utils.data import DataLoader
from detector.CasA.pcdet.config import casa_cfg_from_yaml_file, casa_cfg
from detector.CasA.pcdet.models import casa_build_network
from detector.CasA.pcdet.utils import common_utils
from detector.FasterRCNN.predict import FasterRCNNDetector
from detector.TED.pcdet.config import ted_cfg_from_yaml_file, ted_cfg
from detector.mask_rcnn.predict2 import MaskRCNNDetector
from detector.pcdet.pcdet.config import pcdet_cfg_from_yaml_file, pcdet_cfg
from mot.tracker.TONTDMOT import YONTDMOT
from mot.utils.config import Config
from detector.pcdet.pcdet.models import load_data_to_gpu
from detector.CasA.pcdet.utils.calibration_kitti import Calibration
from dataset_utils.kitti.kitti_tracking_dataset import KittiTrackingDataset
from mot.utils.file_operate import save_results, mkdir_if_inexistence
from mot.tracker.detections import Detections
from evaluation.KITTI.evaluation_HOTA.scripts.run_kitti import eval_kitti
from detector.pcdet.pcdet.models import pcdet_build_network
from mot.utils.file_operate import save_results
from mot.utils.utils import kitti_deconde_dets
from detector.TED.pcdet.models import ted_build_network


def parse_args():
    parser = argparse.ArgumentParser(description="DetMOT")
    parser.add_argument('--cfg_file', type=str, default="configs/kitti.yaml", help='specify the config for demo')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    return args


def track_per_seq(seq_id, model_3d, model_2d, dataloader, cfg_det, cfg_mot, data_path):

    model_3d.cuda()
    model_3d.eval()

    seq_name = str(seq_id).zfill(4)

    tracker = YONTDMOT(cfg_mot)

    out_img_file = os.path.join(cfg_mot.save_path, cfg_mot.split, "image", seq_name); mkdir_if_inexistence(out_img_file)
    out_evl_file = os.path.join(cfg_mot.save_path, cfg_mot.split, "results", "data"); mkdir_if_inexistence(out_evl_file)
    out_evl_txt = os.path.join(out_evl_file, seq_name + ".txt")

    all_time = 0

    dataset = dataloader.dataset
    max_frame = len(dataset)

    for frame, data_dict in enumerate(dataloader):
        print_str = " \r processing %s:  %d/%d" % (seq_name, frame, max_frame)
        sys.stdout.write(print_str)
        sys.stdout.flush()
        trk_data_dict = data_dict['track_result_dict'][0]
        del data_dict['track_result_dict']
        det_data_dict = data_dict
        calib = Calibration(trk_data_dict["calib_path"])
        load_data_to_gpu(det_data_dict)

        det_data_dict_copy = det_data_dict.copy()

        if cfg_mot.public_detector:
            bbox_camera = trk_data_dict["dets_camera"]
            bbox_scores = trk_data_dict["dets_score"]
            bbox_image = trk_data_dict["dets_image"]
            bbox_lidar = trk_data_dict["dets_lidar"]
            bbox_label = trk_data_dict["bbox_label"]
        else:
            pred_dicts, ret_dict, batch_dict = model_3d.forward(det_data_dict)  # (x, y, z, heading, l, h, w) 激光雷达坐标系
            bbox_scores, bbox_lidar, bbox_label, bbox_camera, bbox_image = kitti_deconde_dets(pred_dicts, dataset, batch_dict, cfg_mot.class_names)

        dets = []
        for i in range(len(bbox_scores)):
            if bbox_label[i] == "Car" or bbox_label[i] == "Pedestrian":
                det = Detections(frame, bbox_lidar[i], bbox_scores[i], bbox_label[i], bbox_camera[i], bbox_image=bbox_image[i])
                dets.append(det)

        start = time.time()
        trackers = tracker.update("kitti",
                                  frame,
                                  dets,
                                  cfg_det,
                                  dataset,
                                  det_data_dict_copy,
                                  model_3d,
                                  model_2d,
                                  image_path=trk_data_dict['image_path'],
                                  imu_poses=trk_data_dict['imu_poses'],
                                  calib=calib)
        end = time.time()
        all_time += end - start

        save_results(trackers, trk_data_dict["image"], out_img_file, out_evl_txt, seq_name, frame, trk_data_dict["calib_path"], "Car")
    print(" The sequence process takes of  %f seconds" % all_time)


if __name__ == "__main__":
    try:
        shutil.rmtree("./output")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    args = parse_args()
    cfg_mot = Config(args.cfg_file)

    log_file_path = os.path.join(cfg_mot.save_path, "logs_file")
    os.makedirs(log_file_path, exist_ok=True)
    seq_list = cfg_mot.tracking_seqs

    print("gt seqs: ", seq_list)

    data_path = os.path.join(cfg_mot.data_path, cfg_mot.split)

    start_time = time.time()

    # ------------------------------------------- 3D Detector initialization -------------------------------------
    for id in tqdm.trange(len(seq_list)):
        seq_id = seq_list[id]
        logger = common_utils.create_logger()
        if cfg_mot.detector_name_3d == "CasA":
            casa_cfg_from_yaml_file(cfg_mot.CasA.cfg, casa_cfg)
            cfg_det = casa_cfg
            ckpt = cfg_mot.CasA.ckpt
            det_dataset = KittiTrackingDataset(
                seq_id, dataset_cfg=cfg_det.DATA_CONFIG, class_names=cfg_det.CLASS_NAMES, training=False,
                root_path=data_path, ext=args.ext, logger=logger
            )
            dataloader = DataLoader(det_dataset, batch_size=1, pin_memory=True, collate_fn=det_dataset.collate_batch)
            model_3d = casa_build_network(model_cfg=cfg_det.MODEL, num_class=len(cfg_det.CLASS_NAMES), dataset=det_dataset)

        elif cfg_mot.detector_name_3d == "TED":
            ted_cfg_from_yaml_file(cfg_mot.TED.cfg, ted_cfg)
            cfg_det = ted_cfg
            ckpt = cfg_mot.TED.ckpt
            det_dataset = KittiTrackingDataset(
                seq_id, dataset_cfg=cfg_det.DATA_CONFIG, class_names=cfg_det.CLASS_NAMES, training=False,
                root_path=data_path, ext=args.ext, logger=logger
            )
            dataloader = DataLoader(det_dataset, batch_size=1, pin_memory=True, collate_fn=det_dataset.collate_batch)
            model_3d = ted_build_network(model_cfg=cfg_det.MODEL, num_class=len(cfg_det.CLASS_NAMES),
                                        dataset=det_dataset)

        elif cfg_mot.detector_name_3d == "PCDet":
            pcdet_cfg_from_yaml_file(cfg_mot.PCDet[cfg_mot.pcdet_yaml].cfg, pcdet_cfg)
            cfg_det = pcdet_cfg
            ckpt = cfg_mot.PCDet[cfg_mot.pcdet_yaml].ckpt
            det_dataset = KittiTrackingDataset(
                seq_id, dataset_cfg=cfg_det.DATA_CONFIG, class_names=cfg_det.CLASS_NAMES, training=False,
                root_path=data_path, ext=args.ext, logger=logger
            )
            dataloader = DataLoader(det_dataset, batch_size=1, pin_memory=True, collate_fn=det_dataset.collate_batch)
            model_3d = pcdet_build_network(model_cfg=cfg_det.MODEL, num_class=len(cfg_det.CLASS_NAMES),
                                        dataset=det_dataset)

        model_3d.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)

        # ------------------------------------------- 2D Detector initialization -------------------------------------
        if cfg_mot["detector_name_2d"] == "FasterRCNN":
            model_2d = FasterRCNNDetector(cfg=cfg_mot)
        elif cfg_mot["detector_name_2d"] == "MaskRCNN":
            model_2d = MaskRCNNDetector(cfg=cfg_mot)

        # -------------------------------------------------- Tracking -----------------------------------------------
        track_per_seq(seq_id, model_3d, model_2d, dataloader, cfg_det, cfg_mot, data_path)

    end_time = time.time()
    print("Spend time: %f s" % (end_time-start_time))

    # ------------------------------------------ Starting Evaluation ---------------------------------------------
    print("--------------Starting Evaluation-------------")
    results = eval_kitti()

    # FP = results[0]['Kitti2DBox']['data']['COMBINED_SEQ']['car']['CLEAR']['CLR_FP']
    # FN = results[0]['Kitti2DBox']['data']['COMBINED_SEQ']['car']['CLEAR']['CLR_FN']
    # TP = results[0]['Kitti2DBox']['data']['COMBINED_SEQ']['car']['CLEAR']['CLR_TP']
    # IDSW = results[0]['Kitti2DBox']['data']['COMBINED_SEQ']['car']['CLEAR']['IDSW']
    # MOTA = results[0]['Kitti2DBox']['data']['COMBINED_SEQ']['car']['CLEAR']['MOTA']
    # HOTA = float(results[2][0]['HOTA'])
    # AssA = float(results[2][0]['AssA'])
    # Sum1 = FP + FN + IDSW
    # Sum2 = FP + FN
    # print("age=%d" % confidence_his_max)
    # all_metric = "当max_ages=%d时，HOTA:%.4f, AssA=%.4f, TP:%d, FP:%d, FN:%d, IDSW:%d, MOTA:%.4f, Sum1:%d, Sum2:%d\n" % \
    #              (confidence_his_max, HOTA, AssA, TP, FP, FN, IDSW, MOTA, Sum1, Sum2)
    # save_metric_file_path = open("confidence_his_max.txt", 'a')
    # save_metric_file_path.write(all_metric)
    # save_metric_file_path.close()
