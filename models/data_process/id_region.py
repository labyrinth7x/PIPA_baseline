import os
import shutil
import cv2
import json
import numpy as np


# copy images from images_root to result_root based on the label.
def data_split(images_root, index_root, flag = 0):
    dir_labels = {}
    dir_bboxes = {}
    index_file = open(index_root, "r")
    lines = index_file.readlines()
    for index, line in enumerate(lines):
        #if not os.path.exists(result_root):
            #os.makedirs(result_root)
        dict = line.strip().split()
        label = int(dict[- 2 - flag])
        #if not os.path.isdir(result_root + "/" + str(label)):
            #os.makedirs(result_root + "/" + str(label))
        img_root = dict[0] + "_" + dict[1] + ".jpg"
        #root = result_root + "/" + str(label) + "/" + img_root
        src_root = images_root + "/" + img_root
        #dst_root = result_root + "/" + str(label) + "/" + img_root
        # bbox of the head. (ground truth)
        bbox = [int(dict[2]), int(dict[3]), int(dict[2]) + int(dict[4]), int(dict[3]) + int(dict[5])]
        img = cv2.imread(src_root)
        roi = get_roi(bbox, img)
        if roi is None:
            continue
        if img_root not in dir_labels:
            dir_labels[img_root] = [str(label)]
            dir_bboxes[img_root] = [bbox]
        else:
            dir_labels[img_root].append(str(label))
            dir_bboxes[img_root].append(bbox)
        #shutil.copyfile(src_root, dst_root)
        #if roi is not None:
            #cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,255,255), 2 ,2)
        #cv2.imwrite(dst_root, img)
    return dir_labels, dir_bboxes


def get_dict(dicts_root):
    dicts = []
    for index in range(2):
        f = open(dicts_root[index],"r")
        a = f.read()
        dict = eval(a)
        dicts.append(dict)
        f.close()
    return dicts

# ground truth head --- bbox & labels
def save_dict(dicts_root):
    dir_labels = []
    dir_bboxes = []
    base_root = "/mnt/data-1/data/qi01.zhang/COCO/data/"
    image_folders = ["/mnt/data-1/data/qi01.zhang/PIPA/train","/mnt/data-1/data/qi01.zhang/PIPA/val", "/mnt/data-1/data/qi01.zhang/PIPA/test", "/mnt/data-1/data/qi01.zhang/PIPA/test"]
    index_files = [base_root + "data_split/train/train.txt", base_root + "data_split/val/val.txt",base_root + "data_split/test/test0.txt", base_root + "data_split/test/test1.txt"]
    #result_folders = [base_root + "id/train", base_root + "id/val", base_root + "id/test0", base_root +"id/test1"]
    for index in range(2):
        dir_label, dir_bbox = data_split(image_folders[index], index_files[index],0)
        dir_labels.append(dir_label)
        dir_bboxes.append(dir_bbox)
    for index in range(2,4):
        dir_label, dir_bbox = data_split(image_folders[index], index_files[index],1)
        dir_labels.append(dir_label)
        dir_bboxes.append(dir_bbox)
    f = open(dicts_root[0],"w")
    f.write(str(dir_labels))
    f.close()
    f = open(dicts_root[1],"w")
    f.write(str(dir_bboxes))
    f.close()


# store key info <person,face,head...> into dictionay - <anno_regions>.
def get_anno_dict(anno_file, key):
    anno_regions = {}
    file = open(anno_file,"r")
    for line in file.readlines():
        if line is None:
            break
        info = json.loads(line)
        if key in info:
            image_key = info["image_key"]
            #anno_regions[image_key] = info[key]
            anno_regions[image_key] = line
    return anno_regions


def get_region(dir_labels, dir_bboxes, anno_file, key = "person"):
    count = 0
    anno_regions = get_anno_dict(anno_file, key)
    category = ["train", "val", "test0", "test1"]
    idx = 0
    #anno_result_root = anno_file.split('.json')[0] + '_mod.json'
    #anno_result = open(anno_result_root, 'w')
    # every image in anno_file
    for image in anno_regions:
        line = anno_regions[image]
        info = json.loads(line)
        body_regions = info[key]
        # every body_region in an image for anno
        for inx, body_region in enumerate(body_regions):
            ignore = str(body_region["attrs"]["ignore"])
            occlusion = str(body_region["attrs"]["occlusion"])
            if ignore == "yes":
                info[key][inx]['track_id'] = -1
                continue
            if occlusion == "invisible":
                info[key][inx]['track_id'] = -1
                continue
            bbox = body_region["data"]
            bbox = [int(point) for point in bbox]
            img_name = image
            # determine the category which body_region belongs to 
            for index, dir in enumerate(dir_labels):
                # determine img in train or val or test0 or test1
                if img_name in dir:
                    idx = index
                    if index == 2 or index == 3:
                        img_root = "/mnt/data-1/data/qi01.zhang/PIPA/test/" + img_name
                    else:
                        img_root = "/mnt/data-1/data/qi01.zhang/PIPA/" + category[index] + "/" + img_name
                    break
            img = cv2.imread(img_root)
            roi = get_roi(bbox,img)
            if roi is not None:
                if img_name not in dir_bboxes[idx]:
                    continue
                # visualize region in the images with the ground truth bbox
                head_bboxes = dir_bboxes[idx][img_name]
                max_index = get_label(bbox, head_bboxes)
                if max_index == -1:
                    info[key][inx]['track_id'] = -1
                    continue
                count += 1
                match_label = dir_labels[idx][img_name][max_index]
                # visualize region in the images with the ground truth bbox

                info[key][inx]['track_id'] = match_label
                #result_root = "/mnt/data-1/data/qi01.zhang/COCO/data/" + key + "_anno/data_unprocessed/" + category[idx] + "/" + match_label
                result_root = '/mnt/data-1/data/qi01.zhang/COCO/data/visualize_id_0.35/' + match_label
                img_visualize = cv2.imread(result_root + '/' + img_name)
                cv2.rectangle(img_visualize,(bbox[0], bbox[1]),(bbox[2], bbox[3]), (255,0,255), 2 ,2)
                if not os.path.exists(result_root):
                    os.makedirs(result_root)
                roi = np.array(roi)
                #cv2.imwrite(result_root + "/" + img_name, roi)
                cv2.imwrite(result_root + '/' + img_name, img_visualize)
        #anno_result.write(json.dumps(info))
        #anno_result.write('\n')
    print(count)


def iou_mod(rect1, rect2):
    length = []
    for index in range(2):
        max_length = max(rect1[index], rect2[index])
        min_length = min(rect2[index] + rect1[index + 2], rect2[index] + rect2[index + 2])
        if max_length >= min_length:
            return 0
        length.append(min_length - max_length)
    overlap = length[0] * length[1]
    iou = overlap / ( rect1[2] * rect1[3] + rect2[2] * rect2[3] - overlap)
    return iou


def get_iou(rect1, rect2):
    max_left = max(rect1[0], rect2[0])
    min_right = min(rect1[2], rect2[2])
    width = min_right - max_left
    if width <= 0:
        return 0
    max_up = max(rect1[1], rect2[1])
    min_down = min(rect1[3], rect2[3])
    height = min_down - max_up
    if height <= 0:
        return 0
    overlap = width * height
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    if overlap < 0:
        return 0
    iou = overlap / float(area1 + area2 - overlap)
    return iou


def get_label(region_unlabeled, region_labeled):
    iou = []
    for region in region_labeled:
        iou.append(get_iou(region, region_unlabeled))
    iou_max = max(iou)
    print(iou_max)
    if iou_max < 0.35:
        return -1
    return iou.index(iou_max)


# get head region from the original image
def get_roi(bbox, image):
    roi = []
    up = max(bbox[1],0)
    down = min(bbox[3], image.shape[0])
    left = max(bbox[0], 0)
    right = min(bbox[2], image.shape[1])
    if down <= 0 or right <= 0 or up >= image.shape[0] or left >= image.shape[1]:
        return None
    if up == down or left == right:
        return None
    for index in range(up, down):
        roi.append(image[index][left:right])
    return roi

if __name__=="__main__":
    dict_files = ['/mnt/data-1/data/qi01.zhang/COCO/data/dict/dicts_label.txt','/mnt/data-1/data/qi01.zhang/COCO/data/dict/dicts_bbox.txt']
    head_anno = '/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno/train/head/head.json'
    #save_dict(dict_files)
    dir_labels, dir_bboxes = get_dict(dict_files)
    get_region(dir_labels, dir_bboxes, head_anno, key = 'head')
    #get_region(dir_labels, dir_bboxes, face_annos, "face")
