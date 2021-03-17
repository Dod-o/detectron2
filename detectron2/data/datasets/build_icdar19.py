import cv2
import os
import glob
from tqdm import tqdm
from xml.dom.minidom import parse
import numpy as np
import sys


from detectron2.structures import BoxMode

def get_icard19_dataset(mode, data_dir):
    # # if path == 'train':
    # #     path = '../../dataset/ICDAR2019_cTDaR/training/TRACKA/ground_truth'
    # # elif path == 'val':
    # #     path = '../../dataset/ICDAR2019_cTDaR/test/TRACKA'
    # if mode == 'train':
    #     path = 'data/modern/train'
    # elif mode == 'val':
    #     path = 'data/modern/val'

    ocr_path = os.path.join(data_dir, mode, 'ocr')
    box_path = os.path.join(data_dir, mode, 'bbox')
    img_path = os.path.join(data_dir, mode, 'img')


    # filelist = glob.glob(os.path.join(path, '*.xml'))
    dataset_dicts = []
    filelist1 = glob.glob(os.path.join(img_path, '*.jpg'))
    filelist1.sort()
    filelist2 = glob.glob(os.path.join(img_path, '*.png'))
    filelist3 = glob.glob(os.path.join(img_path, '*.JPG'))
    filelist = filelist1 + filelist2 + filelist3

    filelist = filelist

    for i in tqdm(range(len(filelist))):
        record = {}

        cur_pic_path = filelist[i]
        cur_gt_path = os.path.join(box_path, os.path.basename(filelist[i])[:-4] + '.xml')

        # a = cv2.imread(cur_pic_path)
        height, width = cv2.imread(cur_pic_path).shape[:2]
        record["file_name"] = cur_pic_path
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        xml_tree = parse(cur_gt_path)
        documentElement = xml_tree.documentElement
        # root = xml_tree.getroot()

        tables = documentElement.getElementsByTagName('table')
        objs = []
        for table in tables:
            cur_table = table.getElementsByTagName('Coords')
            assert len(cur_table) == 1
            cur_table = cur_table[0]

            cur_bbox = cur_table.getAttribute('points').split(' ')
            px = [int(item.split(',')[0]) for item in cur_bbox]
            py = [int(item.split(',')[1]) for item in cur_bbox]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]


            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                'text':' this is a sentence.',
                'token': [0, 10, 12, 23, 43]
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts