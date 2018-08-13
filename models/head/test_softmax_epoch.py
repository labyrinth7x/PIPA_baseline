import argparse
from test_softmax import main,get_data,get_matrix_confusion,get_mislabel_list
from process_matrix_confusion import visualize_dict 
import logging
import json

def parse_args():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--gpus', type = str, default='0')
    #parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test0_root', type=str, default ='/mnt/data-1/data/qi01.zhang/COCO/data/head/lst/test0.rec')
    parser.add_argument('--test1_root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/head/lst/test1.rec')
    #parser.add_argument('--lr', type = float,default = 0)
    #parser.add_argument('--wd', type = float,default = 0)
    parser.add_argument('--prefix', type = str)
    parser.add_argument('--mislabel_sort', action='store_true')
    parser.add_argument('--epoch', type = int)
    parser.add_argument('--epoch_begin', type = int ,default = 1)
    #parser.add_argument('--epoch_change', type = int, default = 10)
    #parser.add_argument('--gamma',type = float, default = 0.3)
    parser.add_argument('--interval', type = int, default = 5)
    parser.add_argument('--region', type = str, default = 'head')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--mean_scale',action='store_false')
    parser.add_argument('--person_padding', action = 'store_true')
    parser.add_argument('--euclid',action = 'store_true')
    args = parser.parse_args()
    return args


if __name__ =="__main__":
    args = parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    index = args.prefix.rfind('/')
    index = args.prefix.rfind('/', 0, index)
    param = args.prefix.split('/')[-2]
    result_all_root = args.prefix[:index + 1]
    file_result_all = open(result_all_root + 'out.txt', 'a+')
    result_root = args.prefix[:args.prefix.rfind('/') + 1]
    if args.euclid:
        file_matrix = open(result_root + 'matrix_euclid.json', 'w')
        if args.restart:
            file_result = open(result_root + 'out_euclid.txt', 'w')
        else:
            file_result = open(result_root + 'out_euclid.txt', 'a+')
    else:
        file_matrix = open(result_root + 'matrix.json', 'w')
        if args.restart:
            file_result = open(result_root + 'out.txt', 'w')
        else:
            file_result = open(result_root + 'out.txt', "a+")
    test0,test1 = get_data(args)
    best_ac = 0.0
    matrix_confusion = {}
    mislabel_list = {}
    for epoch in range(args.epoch_begin, args.epoch + 1, args.interval):
        args.test_epoch = epoch
        ac, labels_true, labels_predict, mislabel = main(args, test0, test1)
        if float(ac) > float(best_ac):
            best_ac = float(ac)
            matrix_confusion = get_matrix_confusion(labels_true, labels_predict)
            mislabel_list = mislabel
        logging.info('The accuracy is %f',ac)
        file_result.write('epoch-' + str(epoch) + '-ac-' + str(ac) + '\n')
    file_result.close()
    if args.euclid:
        file_result_all.write(param + '-ac-euclid-' + str(best_ac) + '\n')
    else:    
        file_result_all.write(param + '-ac-' + str(best_ac) + '\n')
    file_result_all.close()
    file_matrix.write(json.dumps(matrix_confusion))
    file_matrix.close()
    # visualize matrix_confusion
    if args.euclid:
        visualize_dict(result_root + 'matrix_euclid.json')
    else:
        visualize_dict(result_root + 'matrix.json')
    # mislabel list
    if args.euclid:
        get_mislabel_list(args, mislabel_list, result_root + 'mislabel_euclid.txt')
    else:
        get_mislabel_list(args, mislabel_list, result_root + 'mislabel.txt')
