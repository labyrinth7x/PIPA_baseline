import os
import shutil
import cv2
import json
import numpy as np


def data_split(images_root, index_root, result_root, flag = 0):
    dir_root = {}
    index_file = open(index_root, "r")
    lines = index_file.readlines()
    for index, line in enumerate(lines):
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        dict = line.strip().split()
        label = int(dict[-2 - flag])
        if not os.path.isdir(result_root + "/" + str(label)):
            os.makedirs(result_root + "/" + str(label))
        img_root = dict[0] + "_" + dict[1] + ".jpg"
        root = result_root + "/" + str(label) + "/" + img_root
        src_root = images_root + "/" + img_root
        dst_root = result_root + "/" + str(label) + "/" + img_root
        bbox = [int(dict[2]), int(dict[3]), int(dict[2]) + int(dict[4]), int(dict[3]) + int(dict[5])]
        img = cv2.imread(src_root)
        roi = get_roi(bbox, img)
        if roi is None:
            continue
        if img_root not in dir_root:
            dir_root[img_root] = [str(dst_root)]
        else:
            dir_root[img_root].append(str(dst_root))
        #shutil.copyfile(src_root, dst_root)
        cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,255,255), 2 ,2)
        cv2.imwrite(dst_root, img)
    return dir_root


def get_dict(dicts_root):
    f = open(dicts_root[index],"r")
    a = f.read()
    dict = eval(a)
    f.close()
    return dict


def get_anno_dict(anno_files, key):
    anno_regions = {}
    for anno_file in anno_files:
        file = open(anno_file,"r")
        for line in file.readlines():
            if line is None:
                break
            info = json.loads(line)
            if key in info:
                image_key = info["image_key"]
                anno_regions[image_key] = info[key]
    return anno_regions


def get_region(dir_root, anno_files, key = "head"):
    anno_regions = get_anno_dict(anno_files, key)
    category = ["train", "val", "test0", "test1"]
    idx = 0
    for image in anno_regions:
        body_regions = anno_regions[image]
        images_root = []
        for dir in dir_root:
            if image in dir:
                images_root = dir[image]
                break
        for image_root in images_root:
            img = cv2.imread(image_root)
            for body_region in body_regions:
                ignore = body_region["attrs"]["ignore"]
                if ignore is "yes":
                    continue
                bbox = body_region["data"]
                bbox = [int(b) for b in bbox]
                occlusion = body_region["attrs"]["occlusion"]
                cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]), (255,255,0), 2 ,2)
                #cv2.putText(img, occlusion , (bbox[0] - 10, bbox[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 1)
            cv2.imwrite(image_root, img)

# get head region from the original image
def get_roi(bbox, image):
    roi = []
    up = max(bbox[1],0)
    down = min(bbox[3], image.shape[0])
    left = max(bbox[0], 0)
    right = min(bbox[2], image.shape[1])
    if down <= 0 or right <= 0 or up >= image.shape[0] or left >= image.shape[1]:
        return None
    for index in range(up, down):
        roi.append(image[index][left:right])
    return roi


if __name__=="__main__":
    #dict_files = ["/data-sdb/qi01.zhang/COCO/data/visualize_id/dicts_label.txt", "/data-sdb/qi01.zhang/COCO/data/visualize_id/dicts_bbox.txt"]
    #base_root = "/data-sdb/qi01.zhang/dataset/pipa-anno-result-raw-json/"
    #body_annos = [base_root + "100340_2/100340_2.json", base_root + "100350_2/100350_2.json"]
    #face_annos = [base_root + "100340_3/100340_3.json", base_root + "100350_3/100350_3.json"]
    #head_annos = [base_root + "100340_1/100340_1.json", base_root + "100350_1/100350_1.json"]
    #dir_root = get_dict(dict_files)
    #get_region(dir_root,body_annos, "person")
    data_split("/mnt/data-1/data/qi01.zhang/PIPA/test0", "/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test0/test0.txt","/mnt/data-1/data/qi01.zhang/COCO/data/head_gt/test0", 1)
