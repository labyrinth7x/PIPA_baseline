# split anno-test-data to test0 and test1

import os
import json
import shutil

# generate test0.json test1.json
def data_split_test(anno_root, query_root, result_root, key = 'head'):
    if not os.path.exists(result_root + '/test0/' + key):
        os.makedirs(result_root + '/test0/' + key)
    if not os.path.exists(result_root + '/test1/' + key):
        os.makedirs(result_root + '/test1/' + key)
    anno_test0 = open(result_root + '/test0/' + key + '/' + key + '.json','w')
    anno_test1 = open(result_root + '/test1/' + key + '/' + key + '.json','w')
    num = 0
    for anno_folder in anno_root:
        folder = os.listdir(anno_folder)
        for anno_file in folder:
            if os.path.isdir(anno_folder + '/' + anno_file):
                continue
            if anno_file.find('.json') == -1:
                continue
            anno = open(anno_folder + '/' + anno_file,'r')
            for line in anno.readlines():
                num += 1
                info = json.loads(line)
                image_key = info['image_key']
                image_test0 = query_root + '/test0/' + image_key
                image_test1 = query_root + '/test1/' + image_key
                if os.path.exists(image_test0):
                    anno_test0.write(line)
                if os.path.exists(image_test1):
                    anno_test1.write(line)


# generate train.json val.json
def data_split(anno_root, result_root, key = 'head', category = 'train'):
    if not os.path.exists(result_root + '/' + category):
        os.mkdirs(result_root + '/' + category)
    anno_result = open(result_root + '/' + category + '/' + key + '/' + key + '.json' , 'w')
    for anno_folder in anno_root:
        folder = os.listdir(anno_folder)
        for anno_file in folder:
            if os.path.isdir(anno_folder + '/' + anno_file):
                continue
            if anno_file.find('.json') == -1:
                continue
            anno = open(anno_folder + '/' + anno_file, 'r')
            for line in anno.readlines():
                anno_result.write(line)


# generate test0 and test1 according to test0.txt test1.txt
def dataset_split(anno_root, images_root):
    if not os.path.exists(images_root + '/test0'):
        os.makedirs(images_root + '/test0')
    if not os.path.exists(images_root + '/test1'):
        os.makedirs(images_root + '/test1')
    for index, anno_file in enumerate(anno_root):
        anno = open(anno_file, 'r')
        for line in anno.readlines():
            dirs = line.split()
            image_key = dirs[0] + '_' + dirs[1] + '.jpg'
            src_root = images_root + '/test/' + image_key
            dst_root = images_root + '/test' + str(index) + '/' + image_key
            if not os.path.exists(dst_root):
                shutil.copyfile(src_root , dst_root)
            




if __name__ == "__main__":
    query_root = '/mnt/data-1/data/qi01.zhang/PIPA'
    result_root = '/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno'
    anno_root = ['/mnt/data-1/data/qi01.zhang/COCO/data/pipa-anno/anno_original/test_val/6031_100002']
    #anno_root = ["/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test0/test0.txt", '/mnt/data-1/data/qi01.zhang/COCO/data/data_split/test1/test1.txt']
    #dataset_split(anno_root,query_root)
    data_split_test(anno_root, query_root, result_root, key = 'face')
