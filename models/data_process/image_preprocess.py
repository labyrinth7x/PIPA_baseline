import cv2
import mxnet as mx
import os
import numpy as np
import argparse

def resize(src_root, dst_root, region):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    folders = os.listdir(src_root)
    for folder in folders:
        dst_folder = dst_root + '/' + folder
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        images = os.listdir(src_root + '/' + folder)
        for image in images:
            img_root = dst_folder + '/' + image
            img = cv2.imread(src_root + '/' + folder + '/' + image)
            if region == 'person':
                cropped_in = cv2.resize(img, (128, 256), interpolation = cv2.INTER_CUBIC)
            elif region == 'face':
                cropped_in = cv2.resize(img, (96, 112), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(dst_folder + '/' + image, cropped_in)

def compute_mean(images_root,region = 'face'):
    folders = os.listdir(images_root)
    sum_blue = 0
    sum_green = 0
    sum_red = 0
    num = 0
    for folder in folders:
        if not os.path.isdir(images_root + '/' + folder):
            continue
        imgs = os.listdir(images_root + '/' + folder)
        # opencv [b,g,r]
        for image in imgs:
            img = cv2.imread(images_root + '/' + folder + '/' + image)
            channel_blue = img[:,:,0]
            channel_green = img[:,:,1]
            channel_red = img[:,:,2]
            sum_green += np.ndarray.sum(channel_green)
            sum_blue += np.ndarray.sum(channel_blue)
            sum_red += np.ndarray.sum(channel_red)
            num += 1
    scale = num
    if region == 'face':
        scale = 112 * 96 * scale
    elif region == 'head':
        scale = scale * 224 *224
    elif region == 'person':
        scale = scale * 256 * 128
    return float(sum_red) / scale, float(sum_green) / scale, float(sum_blue) / scale

def reprocess_person_region(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    # body_shape (560, 224, 3)
    cas = os.listdir(src_root)
    height_max = 560.0
    for ca in cas:
        folders = os.listdir(src_root + '/' + ca)
        for folder in folders:
            img_root = src_root + '/' + ca + '/' + folder
            imgs = os.listdir(img_root)
            for img in imgs:
                img_ori = cv2.imread(img_root + '/' + img)
                height_ori, width_ori, channel = img_ori.shape
                ratio = 224.0 / width_ori
                # resize height to height_max
                if ratio * height_ori > height_max:
                    height_resize = int(height_max)
                    ratio = height_max / height_ori
                    width_resize = int(ratio * width_ori)
                    width_padding_left = int((224.0 - width_resize) / 2)
                    width_padding_right = int(224.0 - width_resize - width_padding_left)
                    height_padding_top = height_padding_bottom = 0
                # resize width to 224
                else:
                    width_resize = 224
                    height_resize = int(ratio * height_ori)
                    width_padding_left = width_padding_right = 0
                    height_padding_top = int((height_max - height_resize) / 2)
                    height_padding_bottom = int(height_max - height_resize - height_padding_top)
                img_resize = cv2.resize(img_ori,(width_resize,height_resize),interpolation = cv2.INTER_AREA)
                # padding - cv2.BORDER_REFLECT
                img_resize = cv2.copyMakeBorder(img_resize, height_padding_top, height_padding_bottom, width_padding_left, width_padding_right, cv2.BORDER_CONSTANT,value=0)
                img_dst_root = dst_root + '/' + ca + '/' + folder
                if not os.path.exists(img_dst_root):
                    os.makedirs(img_dst_root)
                cv2.imwrite(img_dst_root + '/' + img, img_resize)


def parse_args():
    parser = argparse.ArgumentParser(description="command for resize images")
    parser.add_argument('src_root', type = str)
    parser.add_argument('dst_root', type = str)
    parser.add_argument('region', type = str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #args = parse_args()
    #resize(args.src_root, args.dst_root, args.region)
    #images_root = '/mnt/data-1/data/qi01.zhang/COCO/data/head/train/'
    #mean_r, mean_g, mean_b = compute_mean(images_root, region = 'head')
    #print(mean_r)
    #print(mean_g)
    #print(mean_b)
    src_root = '/mnt/data-1/data/qi01.zhang/COCO/data/person_anno/data_unprocessed'
    dst_root = '/mnt/data-1/data/qi01.zhang/COCO/data/person_padding/data_processed'
    reprocess_person_region(src_root, dst_root)
