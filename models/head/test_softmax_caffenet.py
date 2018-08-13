import mxnet as mx
from mxnet import io
import argparse
import logging
import time
import numpy as np
import copy
from utils import point_distances
import os
import cv2

def get_data(args):
    mean_r,mean_g,mean_b,scale = 0.0,0.0,0.0,1.0
    if args.region == 'head':
        shape = (3,224,224)
        resize = 224
        if args.mean_scale:
            mean_r = 66.427663666
            mean_g = 51.5901070721
            mean_b = 44.509282825
            scale = 0.0078125
    elif args.region == 'person':
        if args.person_padding:
            args.test0_root = args.test0_root.replace('head','person_padding')
            args.test1_root = args.test1_root.replace('head','person_padding')
            shape = (3,560,224)
        else:
            args.test0_root = args.test0_root.replace('head', 'person_anno')
            args.test1_root = args.test1_root.replace('head', 'person_anno')
            shape = (3,227,227)
        resize = 227
        if args.mean_scale:
            mean_r = 109.331541373
            mean_g = 94.3656742673
            mean_b = 88.0521883656
            scale = 0.0078125
    elif args.region == 'face':
        shape = (3,112,96)
        resize = -1
        #shape = (3,28,28)
        #resize = 28
        args.test0_root = args.test0_root.replace('head','face_anno')
        args.test1_root = args.test1_root.replace('head','face_anno') 
        if args.mean_scale:
            mean_r = 138.53685211
            mean_g = 100.649704666
            mean_b = 85.3696663188
            scale = 0.0078125
    test0_data = mx.io.ImageRecordIter(
        path_imgrec = args.test0_root,
        batch_size = 1,
        label_name = "label",
        data_name = "data",
        #round_batch = 0,
        resize = resize,
        data_shape = shape,
        mean_r = mean_r,
        mean_g = mean_g,
        mean_b = mean_b,
        scale = scale
    )
    test1_data = mx.io.ImageRecordIter(
        path_imgrec = args.test1_root,
        batch_size = 1,
        data_name = "data",
        label_name="label",
        # round_batch = 0,
        resize = resize,
        data_shape = shape,
        mean_r = mean_r,
        mean_g = mean_g,
        mean_b = mean_b,
        scale = scale
    )
    return test0_data,test1_data
 
def get_network(args):
    symbol, arg_params,aux_params = mx.model.load_checkpoint(args.prefix, args.test_epoch)
    all_layers = symbol.get_internals()
    net = all_layers['embed_output']
    #net = all_layers['flatten0_output']
    return net, arg_params, aux_params

def main(args, test0, test1):
    net, arg_params, aux_params = get_network(args)
    # net = mx.sym.L2Normalization(data=net,mode='instance')
    # net = mx.sym.FullyConnected(data = net ,num_hidden = 256,name = 'fc_1')
    # net = mx.sym.Activation(data = net, act_type = 'relu')
    # net = mx.sym.FullyConnected(data=net,num_hidden = 2356,name='fc')
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    mod = mx.mod.Module(symbol = net, context=devs,label_names = None)
    mod.bind(for_training=False,data_shapes = test0.provide_data)
    mod.set_params(arg_params, aux_params)
    labels = []
    outputs = mod.predict(test0)
    if args.euclid:
        outputs = outputs.asnumpy()
    else:
        outputs = mx.nd.L2Normalization(outputs, mode ='instance')
        outputs = outputs.asnumpy()
        outputs = outputs.transpose()
    test0.reset()
    for batch in test0:
        label = batch.label[0]
        labels.append(label.asscalar())
    ac = 0.0
    mod.bind(for_training=False,data_shapes = test1.provide_data,force_rebind = True)
    mod.set_params(arg_params, aux_params)
    outputs_test = mod.predict(test1) 
    if args.euclid:
        outputs_test = outputs_test.asnumpy()
        sims = point_distances(outputs_test,outputs)
    else:
        outputs_test = mx.nd.L2Normalization(outputs_test,mode='instance')
        outputs_test = outputs_test.asnumpy()
        sims = np.dot(outputs_test,outputs)
    #sims = point_distances(outputs_test,outputs)
    test1.reset()
    num = 0
    labels_true = []
    labels_predict = []
    mislabel = {}
    inx = 0
    for batch in test1:
        label = batch.label[0].asscalar()
        if args.euclid:
            inx = np.argmin(sims[num])
        else:
            inx = np.argmax(sims[num])
        labels_true.append(int(label))
        labels_predict.append(int(labels[inx]))
        if labels[inx] == label:
            ac += 1
        else:
            mislabel[int(num)] = int(inx)
        num += 1
        #print(ac)
        #print("true label %f" % label)
        #print("predict label %f" % labels[armax])
    return ac / num, labels_true, labels_predict, mislabel

def get_matrix_confusion(labels_true, labels_predict):
    matrix_confusion = {}
    for index in range(len(labels_true)):
        label_true = labels_true[index]
        label_predict = labels_predict[index]
        if label_true in matrix_confusion:
            if label_predict in matrix_confusion[label_true]:
                matrix_confusion[label_true][label_predict] += 1
            else:
                matrix_confusion[label_true][label_predict] = 1
        else:
            matrix_confusion[label_true] = {label_predict:1}
    return matrix_confusion

def get_mislabel_list(args, mislabel, mislabel_root):
    mislabel_file = open(mislabel_root, 'w')
    test0_file = open(args.test0_root.split('rec')[0] + 'lst', 'r')
    test1_file = open(args.test1_root.split('rec')[0] + 'lst', 'r')
    test0_lines = test0_file.readlines()
    test1_lines = test1_file.readlines()
    inx = 0
    for test1_index in mislabel:
        mislabel_file.write(test1_lines[test1_index].split()[2] + ' ' + test0_lines[mislabel[test1_index]].split()[2] + '\n')
    mislabel_file.close()


    

def parse_args():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type = str, default='0')
    parser.add_argument('--test0_root', type=str, default ='/mnt/data-1/data/qi01.zhang/COCO/data/lst_origin/test0.rec')
    parser.add_argument('--test1_root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/lst_origin/test1.rec')
    parser.add_argument('--lr', type = float)
    parser.add_argument('--wd', type = float)
    parser.add_argument('--prefix', type = str)
    parser.add_argument('--test_epoch', type = int)
    parser.add_argument('--epoch_change', type = int, default = 10)
    parser.add_argument('--gamma',type = float, default = 0.3)
    parser.add_argument('--region', type = str, default = 'head')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ac, labels_true, labels_predict = main(args)
    logging.info('The accuracy is %f', ac)
