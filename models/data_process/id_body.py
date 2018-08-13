import os
import shutil
import cv2
import json


# According to index_root, classify images based on the label. 
# dir_labels and dir_bbox restore the labels and bbox info of the same category - train, val, test0, test1
def data_split(images_root, index_root, result_root, flag = 0):
    dir_labels = {}
    dir_bbox = {}
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
        src_root = images_root + "/" + img_root
        dst_root = result_root + "/" + str(label) + "/" + img_root
        # bbox of the ground truth(head)
        bbox = [int(dict[2]), int(dict[3]), int(dict[2]) + int(dict[4]), int(dict[3]) + int(dict[5])]
        if img_root not in dir_labels:
            dir_labels[img_root] = [str(label)]
            dir_bbox[img_root] = [bbox]
        else:
            dir_labels[img_root].append(str(label))
            dir_bbox[img_root].append(bbox)
        shutil.copyfile(src_root, dst_root)
        img = cv2.imread(src_root)
        roi = get_roi(bbox, img)
        #if roi is not None:
            #cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,255,255), 2 ,2)
        #cv2.imwrite(dst_root, img)
    return dir_labels, dir_bbox


def get_region(dirs, anno_files, key = "body"):
    category = ["train", "val", "test0", "test1"]
    for anno_file in anno_files:
        file = open(anno_file,"r")
        for line in file.readlines():
            if line is None:
                break
            info = json.loads(line)
            if "person" in info:
                pers = info["person"]
                # get body from every pic
                for per in pers:
                    bbox = per["data"]
                    bbox = [int(point) for point in bbox]
                    img_name = info["image_key"]
                    for index, dir in enumerate(dirs):
                        # determine img in train or val or test0 or test1
                        if img_name in dir:
                            for label in dir[img_name]:
                                img_root = "/data-sdb/qi01.zhang/COCO/data/id/" + category[index] + "/" + label + "/" + img_name
                                img = cv2.imread(img_root)
                                roi = get_roi(bbox, img)
                                if roi is not None:
                                    cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,255,0),2,2)
                                else:
                                    print("roi is none")
                                cv2.imwrite(img_root, img)
                                                                         
                                                                         
                                                                         
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

def get_label(unlabeled, labeled):
    iou = []
    for head in unlabeled:
        iou.append(iou_mod(head, labeled))
    return iou.index(max(iou))


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
    dirs = []
    base_root = "/data-sdb/qi01.zhang/COCO/data/"
    image_folders = ["/data-sdb/qi01.zhang/PIPA/train","/data-sdb/qi01.zhang/PIPA/val", "/data-sdb/qi01.zhang/PIPA/test", "/data-sdb/qi01.zhang/PIPA/test"]
    index_files = [base_root + "data_split/train/train.txt", base_root + "data_split/val/val.txt",base_root + "data_split/test/test0.txt", base_root + "data_split/test/test1.txt"]
    result_folders = [base_root + "id/train", base_root + "id/val", base_root + "id/test0", base_root +"id/test1"]
    for index in range(2):
        dir = data_split(image_folders[index], index_files[index], result_folders[index])
        dirs.append(dir)
    for index in range(2,4):
        dir = data_split(image_folders[index], index_files[index], result_folders[index], 1)
        dirs.append(dir)
    print("dir done")
    base_root = "/data-sdb/qi01.zhang/dataset/pipa-anno-result-raw-json/"
    body_annos = [base_root + "100340_2/100340_2.json", base_root + "100350_2/100350_2.json"]
    face_annos = [base_root + "100340_3/100340_3.json", base_root + "100350_3/100350_3.json"]
    head_annos = [base_root + "100340_1/100340_1.json", base_root + "100350_1/100350_1.json"]
    get_region(dirs, body_annos, "body")
    get_region(dirs, face_annos, "face")
    get_region(dirs, head_annos, "head")
