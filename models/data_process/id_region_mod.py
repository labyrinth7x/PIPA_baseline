import os
import shutil
import cv2
import json
import numpy as np

def load_dict_head(head_root):
    head_file = open(head_root, 'r')
    lines = head_file.readlines()
    dir_indexes = {}
    dir_bboxes = {}
    for line in lines:
        if not line:
            break
        info = json.loads(line)
        img_key = info['image_key']
        if 'head' not in info:
            continue
        regions = info['head']
        bboxes = []
        indexes = []
        for index,region in enumerate(regions):
            track_id = region['track_id']
            if track_id == -1:
                continue
            bbox = region['data']
            bbox = [int(point) for point in bbox]
            bboxes.append(bbox)
            indexes.append(int(track_id))
        if len(bboxes) != 0:
            dir_bboxes[img_key] = bboxes
            dir_indexes[img_key] = indexes
    return dir_bboxes, dir_indexes
        

def data_split(images_root, anno_root, key = 'head'):
    count_miss = 0
    dir_bboxes = {}
    dir_lines = {}
    dir_indexes = {}
    anno_file = open(anno_root,'r')
    lines = anno_file.readlines()
    for line in lines:
        if not line:
            break
        info = json.loads(line)
        img_key = info['image_key']
        dir_lines[img_key] = line.strip()
        if key not in info:
            continue
        regions = info[key]
        bboxes = []
        indexes = []
        img_root = images_root + '/' + img_key
        img = cv2.imread(img_root)
        for index, region in enumerate(regions):
            ignore = str(region["attrs"]["ignore"])
            occlusion = str(region["attrs"]["occlusion"])
            if ignore == "yes":
                continue
            if occlusion == "invisible":
                continue
            bbox = region['data']
            bbox = [int(point) for point in bbox]
            roi = get_roi(bbox, img)
            if roi is None:
                continue
            indexes.append(index)
            bboxes.append(bbox)
        if len(bboxes) != 0:
            dir_bboxes[img_key] = bboxes
            dir_indexes[img_key] = indexes
        else:
            count_miss += 1
    print(count_miss)
    return dir_bboxes, dir_lines, dir_indexes
            


def get_dicts(dicts_root):
    dicts = []
    for index in range(3):
        f = open(dicts_root[index],"r")
        a = f.read()
        dict = eval(a)
        dicts.append(dict)
        f.close()
    return dicts


def get_region(dir_bboxes, dir_lines, dir_indexes, anno_root, ca = 'train'):
    anno_result = open('/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno/' + ca + '/head/head_mod.json','w')
    if ca == 'train':
        flag = 0
    elif ca == 'val':
        flag = 1
    elif ca == 'test0':
        flag = 2
    else:
        flag = 3
    count_miss = 0
    anno_file = open(anno_root, 'r')
    lines = anno_file.readlines()
    for line in lines:
        dic = line.strip().split()
        label = int(dic[- 2 - flag / 2])
        img_key = dic[0] + "_" + dic[1] + ".jpg"
        bbox = [int(dic[2]), int(dic[3]), int(dic[2]) + int(dic[4]), int(dic[3]) + int(dic[5])]
        img_root = '/mnt/data-1/data/qi01.zhang/PIPA/' + ca + '/' + img_key
        img = cv2.imread(img_root)
        roi = get_roi(bbox,img)
        if roi is None or img_key not in dir_bboxes[flag]:
            count_miss += 1
            continue
        bboxes_unassigned = dir_bboxes[flag][img_key]
        max_index = get_label(bboxes_unassigned, bbox)
        if max_index == -1:
            count_miss += 1
            continue
        result_root = '/mnt/data-1/data/qi01.zhang/COCO/data/head_anno/data_unprocessed/' + ca + '/' + str(label)
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        roi_result = get_roi(bboxes_unassigned[max_index], img)
        roi_result = np.array(roi_result)
        cv2.imwrite(result_root + '/' + img_key, roi_result)
        inx = dir_indexes[flag][img_key][max_index]
        info = json.loads(dir_lines[flag][img_key])
        info['head'][inx]['track_id'] = label
        dir_lines[flag][img_key] = json.dumps(info)
    print(count_miss)
    write_dicts(dir_lines[flag], ca)


def match_head_region_anno(anno_root,result_root, ca = 'train', key = 'person'):
    count = 0
    images_root = '/mnt/data-1/data/qi01.zhang/PIPA/' + ca
    anno_head = anno_root + '/' + ca + '/head/head_mod.json'
    anno_region = anno_root + '/' + ca + '/' + key + '/' + key + '.json'
    anno_result = open(anno_region.split('.json')[0] + '_mod.json','w')
    head_bboxes, head_indexes = load_dict_head(anno_head)
    anno_file = open(anno_region, 'r')
    lines = anno_file.readlines()
    for line in lines:
        if not line:
            break
        info = json.loads(line)
        img_key = info['image_key']
        if key not in info or img_key not in head_bboxes:
            anno_result.write(json.dumps(info))
            anno_result.write('\n')
            continue
        bboxes  = head_bboxes[img_key]
        regions = info[key]
        for inx, region in enumerate(regions):
            ignore = str(region['attrs']['ignore'])
            occlusion = str(region['attrs']['occlusion'])
            if ignore == 'yes' or occlusion == 'yes':
                info[key][inx]['track_id'] = -1
                continue
            bbox = region['data']
            bbox = [int(point) for point in bbox]
            img = cv2.imread(images_root + '/' + img_key)
            roi = get_roi(bbox, img)
            if roi is None:
                info[key][inx]['track_id'] = -1
                continue
            index_max = get_label(bboxes, bbox)
            if index_max == -1:
                info[key][inx]['track_id'] = -1
                continue
            count += 1
            label_match = head_indexes[img_key][index_max]
            info[key][inx]['track_id'] = label_match
            roi = np.array(roi)
            image_result_root = result_root + '/' + key + '_anno/data_unprocessed/' + ca + '/'+ str(label_match)
            if not os.path.exists(image_result_root):
                os.makedirs(image_result_root)
            cv2.imwrite(image_result_root + '/' + img_key, roi)
        anno_result.write(json.dumps(info))
        anno_result.write('\n')
    print(count)
    

 


def save_dict(dicts_root):
    dir_bboxes = []
    dir_lines = []
    dir_annos = []
    base_root = "/mnt/data-1/data/qi01.zhang/COCO/data/"
    image_folders = ["/mnt/data-1/data/qi01.zhang/PIPA/train","/mnt/data-1/data/qi01.zhang/PIPA/val", "/mnt/data-1/data/qi01.zhang/PIPA/test0", "/mnt/data-1/data/qi01.zhang/PIPA/test1"]
    anno_files = [base_root + "pipa-anno/train/head/head.json", base_root + "pipa-anno/val/head/head.json",base_root + "pipa-anno/test0/head/head.json", base_root + "pipa-anno/test1/head/head.json"]
    result_folders = [base_root + "id/train", base_root + "id/val", base_root + "id/test0", base_root +"id/test1"] 
    for index in range(2):
        dir_bbox, dir_line, dir_index = data_split(image_folders[index], anno_files[index])
        dir_bboxes.append(dir_bbox)
        dir_lines.append(dir_line)
        dir_indexes.append(dir_index)
    for index in range(2,4):
        dir_bbox, dir_line, dir_index = data_split(image_folders[index], anno_files[index])
        dir_bboxes.append(dir_bbox)
        dir_lines.append(dir_line)
        dir_indexes.append(dir_index)
    f = open(dicts_root[0],"w")
    f.write(str(dir_bboxes))
    f.close()
    f = open(dicts_root[1],'w')
    f.write(str(dir_lines))
    f.close()
    f = open(dicts_root[2],'w')
    f.write(str(dir_indexes))
    f.close()
    
    
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
    iou = overlap / float(min(area1,area2))
    return iou


def get_label(region_gallery, region_query):
    iou = []
    for region in region_gallery:
        iou.append(get_iou(region, region_query))
    iou_max = max(iou)
    if iou_max < 0.8:
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



def write_dicts(dicts, ca):
    anno_result = open('/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno/' + ca + '/head/head_mod.json','w')
    for key in dicts:
        anno_result.write(dicts[key])
        anno_result.write('\n')

if __name__ =="__main__":
    base_root = '/mnt/data-1/data/qi01.zhang/COCO/data/'
    dict_files = [base_root + 'dict/head_bbox.txt', base_root + 'dict/head_lines.txt', base_root + 'dict/head_indexes.txt']
    ca = 'test0'
    gt_anno = base_root + 'data_split/' + ca + '/' + ca + '.txt'
    #save_dict(dict_files)
    #dir_bboxes, dir_lines, dir_indexes = get_dicts(dict_files)
    #dir_lines = get_region(dir_bboxes, dir_lines, dir_indexes, gt_anno, ca = ca)
    anno_root = '/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno'
    result_root = '/mnt/data-1/data/qi01.zhang/COCO/data/'
    match_head_region_anno(anno_root, result_root, ca = ca, key = 'person')
