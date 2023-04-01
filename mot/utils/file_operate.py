# This code is borrowed from AB3DMOT(https://github.com/xinshuoweng/AB3DMOT)

import copy, os
import numpy as np
from mot.utils.visualization_2d import  show_image_with_boxes_2d
from mot.utils.visualization_3d import show_image_with_boxes_3d


def mkdir_if_inexistence(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def fileparts(input_path, warning=True, debug=True):
    '''
    this function return a tuple, which contains (directory, filename, extension)
    if the file has multiple extension, only last one will be displayed

    parameters:
        input_path:     a string path

    outputs:
        directory:      the parent directory
        filename:       the file name without extension
        ext:            the extension
    '''
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0: return ('', '', '')
    if good_path[-1] == '/':
        if len(good_path) > 1:
            return (good_path[:-1], '', '')  # ignore the final '/'
        else:
            return (good_path, '', '')  # ignore the final '/'

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]  # splitext用于分离文件名与扩展名
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''

    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)  #加入safe_data='/home//user/Documnets'，那么 os.path.normpath(safe_data) = '\home\user\Documnets'
    return safe_data


def save_results(trackers, image, out_img_file, out_evl_txt, seq_name, frame, calib_file, category):
    f = open(out_evl_txt, 'a')

    if len(trackers) > 0:
        for d in trackers:
            tracker = d.flatten()
            bbox_camera = list(map(float,tracker[1:8]))  # 3D bounding box(h,w,l,x,y,z,theta)
            id_tmp = int(tracker[0])
            alpha = 0
            bbox_label = tracker[8]
            bbox_image =  list(map(float,tracker[9:13]))
            conf_tmp = int(tracker[13])
            label = f'{id_tmp} {bbox_label}'
            color = compute_color_for_id(id_tmp)
            # with open(save_name, 'a') as f:
            str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (
                    frame, id_tmp, bbox_label, alpha, bbox_image[0],
                    bbox_image[1], bbox_image[2], bbox_image[3], bbox_camera[0], bbox_camera[1], bbox_camera[2],
                    bbox_camera[3],
                    bbox_camera[4], bbox_camera[5], bbox_camera[6], conf_tmp)
            f.write(str_to_srite)
            img_id = str(frame).zfill(6)
            # show_image_with_boxes_3d(image, bbox_camera, out_img_file, color, img_id, label, calib_file, line_thickness=2)
            # show_image_with_boxes_2d(bbox_image, image, out_img_file, color, img_id, label, line_thickness=2)
        f.close()