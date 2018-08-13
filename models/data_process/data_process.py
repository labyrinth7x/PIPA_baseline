from mxnet import gluon,nd
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import vision
import numpy as np
import cv2
import os
import math
import json

# Use this to encapsulate data when categories are not numbers
# subset_root is the root folder location of images.
def data_process(subset_root, batch_size = 10, shuffle = True):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224)
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
        ])
    loader = gluon.data.DataLoader
    subset_ds = vision.ImageFolderDataset(subset_root, flag = 1)
    # test_ds = SimpleDataset(load_data(test_root, test_index))
    subset_data = loader(subset_ds.transform_first(transform_train), batch_size, shuffle = shuffle, last_batch = "keep")
    # print("train data load done")
    # test_data = loader(test_ds.transform_first(transform_test), batch_size, shuffle = False, last_batch = "keep")
    # return train_data, test_data
    return subset_data
    
# flag is used to separate train/val set from test set.
# image_root is the location of the unarchived images.(train, test, val)
# Crop head from the original image, archive to each folder.
def data_split(images_root, index_root, result_root, flag = 0):
    index_file = open(index_root, "r")
    lines = index_file.readlines()
    index_delete = []
    for index, line in enumerate(lines):
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        dict = line.strip().split()
        label = int(dict[-2 - flag])
        if not os.path.isdir(result_root + "/" + str(label)):
            os.makedirs(result_root + "/" + str(label))
        img_root = dict[0] + "_" + dict[1] + ".jpg"
        root = result_root + "/" + str(label) + "/" + img_root
        img = cv2.imread(images_root + "/" + img_root)
        bbox = [int(dict[2]), int(dict[3]), int(dict[2]) + int(dict[4]), int(dict[3]) + int(dict[5])]
        roi = get_roi(bbox, img)
        if roi is None:
            index_delete.append(index)
            print(line)
            continue
        roi = np.array(roi)
        cv2.imwrite(root,roi)
    index_file.close()
    file = open(index_root.split(".")[0] + "_mod.txt","w")
    index = 0
    for new_index, line in enumerate(lines):
        if index >= len(index_delete):
            if new_index == len(lines) - 1:
                file.write(line.strip())
            else:
                file.write(line)
            continue
        if index_delete[index] == new_index:
            del(lines[new_index])
        else:
            if new_index == len(lines) - 1:
                file.write(line.strip())
            else:
                file.write(line)
    file.close()
   
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
    
    
# subset_id 0 for leftover, 1 for train, 2 for validation, 3 for test
# if subset_id is 3 for test, split_id is 0 for test0, 1 for test1; otherwise, split_id is None
# Use to generate index file for specified (subset_id, split_id).
# generate test0.txt test1.txt according to split_test_original.txt
def index_split(index_root, subset_root, subset_id, split_id = None):
    f = open(index_root,"r")
    file = open(subset_root,"w")
    index = 0
    for line in f.readlines():
        line = line.strip()
        if split_id is not None:
            subset = int(line.split()[-2])
            split = int(line.split()[-1])
        else:
            subset = int(line.split()[-1])
            split = None
        if subset == subset_id:
            if split is not None:
                if split != split_id:
                    continue
            file.write(line + "\n")
            index += 1
    f.close()
    file.close()
    print(index)

def drawFrame(file, rect):
    img = cv2.imread(file)
    cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (255,0,0), 2, 2)
    cv2.imshow("aaa",img)

def iou(rect1, rect2):
    max_left = max(rect1[0], rect2[0])
    min_right = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    if max_left >= min_right:
        return 0
    width = min_right - max_left
    max_up = max(rect1[1], rect2[1])
    min_down = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    if max_up >= min_down:
        return 0
    height = min_down - max_up
    overlap = width * height
    iou = overlap / (rect1[2] * rect1[3] + rect2[2] * rect2[3] - overlap)

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


def find_label(patch_unlabeled, patch_labeled):
    for unlabeled in patch_unlabeled:
        iou = []
        for labeled in patch_labeled:
            iou.append(iou_mod(unlabeled['rect'], labeled['rect']))
        index = iou.index(max(iou))
        # ignore the situation that two faces share the maximum iou with the same head
        unlabeled['id'] = patch_labeled[index]['id']


# return 4 dicts train, val, test0, test1
# dict[file_name] = file_location
def image_dict(files):
    dicts = []
    for index in range(len(files)):
        file = open(files[index],"r")
        dict = {}
        dir = ['train','val','test0','test1']
        for line in file.readlines():
            line = line.strip().split()
            if not line:
                break
            image_path = '/data-sdb/qi01.zhang/PIPA/' + dir[index] + "/" +  line[0] + "_" + line[1] + ".jpg"
            dict[line[0] + "_" + line[1] + ".jpg"] = image_path
        dicts.append(dict)
    return dicts


            
    

# for skeleton
def body_extract(files_in, files_anno):
    dicts = image_dict(files_in)
    for file_in in files_anno:
        file = open(file_in,"r")
        info = file.readline()
        while True:
            data = json.loads(info)
            if "person" in data:
                bbox = data["person"][0]["data"]
                bbox = [int(point) for point in bbox]
                image_root = data["image_key"]
                for index, dict in enumerate(dicts):
                    folder_dir = '/data-sdb/qi01.zhang/COCO/data/body/' + category[index]
                    if image_root in dict:
                        if not os.path.exists(folder_dir):
                            os.mkdir(folder_dir)
                        img = cv2.imread(dict[image_root])
                        # drawFrame(dict[image_root], bbox)
                        cv2.waitKey(10000)
                        roi = get_roi(bbox, img)
                        if roi is None:
                            continue
                        roi = np.array(roi)
                        cv2.imwrite(folder_dir + '/' + image_root,roi)
                        #files_out[index].write(
            info = file.readline()





    

    

if __name__ == "__main__":
    #files_in = ['/data-sdb/qi01.zhang/COCO/data/data_split/train/train.txt','/data-sdb/qi01.zhang/COCO/data/data_split/val/val.txt',
    #'/data-sdb/qi01.zhang/COCO/data/data_split/test/test0.txt','/data-sdb/qi01.zhang/COCO/data/data_split/test/test1.txt']
    #files_anno = ['/data-sdb/qi01.zhang/dataset/pipa-anno-result-raw-json/100340_2/100340_2.json', '/data-sdb/qi01.zhang/dataset/pipa-anno-result-raw-json/100350_2/100350_2.json']
    #body_extract(files_in, files_anno)
    #index_split("/data/public/PIPA/annotations/index.txt","/home/zhangqi/COCO/data/data_split/train/train.txt",1)
    #index_split("/mnt/data-1/data/qi01.zhang/PIPA/annotations/index.txt", "/mnt/data-1/data/qi01.zhang/COCO/data/data_split/val/val.txt",2)
    index_split("/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test0/split_test_original.txt","/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test0/test0.txt",3,0)
    index_split("/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test0/split_test_original.txt","/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test1/test1.txt",3,1)
    # data_split("/mnt/data-1/data/qi01.zhang/PIPA/train", "/mnt/data-1/data/qi01.zhang/COCO/data/data_split/train/train.txt","/mnt/data-1/data/qi01.zhang/COCO/head/train",0)
    #data_split("/mnt/data-1/data/qi01.zhang/PIPA/val", "/mnt/data-1/data/qi01.zhang/COCO/data/data_split/val/val.txt", "/mnt/data-1/data/qi01.zhang/COCO/data/head/val",0)
