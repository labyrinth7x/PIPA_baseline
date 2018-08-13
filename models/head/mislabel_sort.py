import os
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='mislabel pic sort in a folder')
    parser.add_argument('--result_root', type = str, default = '/mnt/data-1/data/qi01.zhang/COCO/model_data/mislabel_data')
    parser.add_argument('--mislabel_file', type = str, default = '/mnt/data-1/data/qi01.zhang/COCO/models/head/mislabel.txt')
    parser.add_argument('--image_root', type = str, default = '/mnt/data-1/data/qi01.zhang/COCO/data/')
    parser.add_argument('--lst_root', type = str, default = '/mnt/data-1/data/qi01.zhang/COCO/data/lst/')
    args = parser.parse_args()
    return args


def mislabel_sort(args):
    if not os.path.exists(args.result_root):
        os.makedirs(args.result_root)
    mislabel_file = open(args.mislabel_file, 'r')
    lines_test0 = open(args.lst_root + 'test0.lst').readlines()
    lines_test1 = open(args.lst_root + 'test1.lst').readlines()
    for idx, line in enumerate(mislabel_file.readlines()):
        if line is None:
            break
        info = line.split()
        index_test1 = int(info[3])
        index_test0 = int(info[7])
        label_test1 = int(info[1])
        label_test0 = int(info[5])
        img_test1 = lines_test1[index_test1].split()[2]
        img_test0 = lines_test0[index_test0].split()[2]
        shutil.copyfile(args.image_root + 'test1/' + img_test1, args.result_root + '/' + img_test1.split('/')[1])
        shutil.copyfile(args.image_root + 'test0/' + img_test0, args.result_root + '/' + img_test0.split('/')[1]
 




if __name__ == "__main__":
    args = parse_args()
    mislabel_sort(args)
