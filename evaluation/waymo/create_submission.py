from evaluation.waymo.waymo import label_pb2
from evaluation.waymo.waymo.protos import metrics_pb2
from evaluation.waymo.waymo.protos import submission_pb2
import pickle
import os
import numpy as np

WAYMO_CLASSES = {'unknown':0, 'Vehicle':1, 'Pedestrian':2, 'Sign':3, 'Cyclist':4}


def converter(dataset_info_path,prediction_info_path):
    data_path=dataset_info_path
    pred_path=prediction_info_path

    import tqdm
    all_objects=[]

    with open(data_path, 'rb') as f:
        data_pickle_file = pickle.load(f)
    with open(pred_path, 'rb') as f:
        pred_pickle_file = pickle.load(f)

    for frame_id in tqdm.trange(len(data_pickle_file)):
        data_frame=data_pickle_file[frame_id]['metadata']
        context_name = data_frame['context_name']
        frame_timestamp_micros = data_frame['timestamp_micros']

        annotations=pred_pickle_file[frame_id]

        obj_name = annotations['name']
        pred_boxes = annotations['boxes_lidar']
        pred_boxes = pred_boxes.astype(np.float16)
        scores = annotations['score']

        if 'obj_ids' in annotations.keys():
            obj_ids = annotations['obj_ids']
        else:
            obj_ids = None

        for i in range(len(obj_name)):
            ob_name=obj_name[i]
            score = scores[i]
            pred_boxe = pred_boxes[i]

            if obj_ids is not None:
                id = str(obj_ids[i])
            else:
                id = '0'

            o = metrics_pb2.Object()
            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros
            box = label_pb2.Label.Box()
            box.center_x = pred_boxe[0]
            box.center_y = pred_boxe[1]
            box.center_z = pred_boxe[2]
            box.length = pred_boxe[3]
            box.width = pred_boxe[4]
            box.height = pred_boxe[5]
            box.heading = pred_boxe[6]
            o.object.box.CopyFrom(box)
            o.score = score
            o.object.id = id
            o.object.type = WAYMO_CLASSES[ob_name]
            all_objects.append(o)

    return all_objects


def creat_submission(dataset_info_path,prediction_info_path,submission_path):
    if not os.path.exists(prediction_info_path):
        return
    submission=submission_pb2.Submission()

    objects = converter(dataset_info_path,prediction_info_path)

    submission.task = submission_pb2.Submission.DETECTION_3D
    name = "**@gmail.com"  # set to your own email
    if name == "**@gmail.com":
        print('please set to your own email !!!')
    print('current email: ', name)
    submission.account_name = name
    submission.unique_method_name = 'YONTD_MOT'
    submission.authors.append('TBD')
    submission.affiliation='TBD'
    submission.description='TBD'
    submission.method_link='TBD'
    submission.sensor_type=submission_pb2.Submission.SensorType.LIDAR_ALL
    submission.number_past_frames_exclude_current=4
    submission.number_future_frames_exclude_current=0

    submission.inference_results.objects.extend(objects)
    submission.latency_second=0

    out_path=submission_path

    f = open(out_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()


def main():
    test_dataset_info_path = 'data/waymo/waymo_infos_test.pkl'
    test_prediction_info_path = "output/result_dets.pkl"
    test_submission_path = 'output/'
    if not os.path.exists(test_submission_path):
        os.makedirs(test_submission_path)
    test_submission_path+='/submission.bin'

    creat_submission(test_dataset_info_path,test_prediction_info_path,test_submission_path)
    # creat_submission(val_dataset_info_path,val_prediction_info_path,val_submission_path)

if __name__ == '__main__':
  main()
