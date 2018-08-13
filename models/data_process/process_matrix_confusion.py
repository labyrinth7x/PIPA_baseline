import os
import argparse


def load_class(images_root):
    classes = []
    folders = os.listdir(images_root)
    folders.sort()
    for folder in folders:
        classes.append(int(folder))
    return classes


def dict2array(dict_root, classes):
    f = open(dict_root, 'r')
    a = f.read()
    dicts = eval(a)
    result_file = open(dict_root.split('.txt')[0] +'_array.txt', 'w')
    classes_vis = [' ' + str(ele) for ele in classes]
    result_file.write(''.join(classes_vis) + '\n')
    for index, class1 in enumerate(dicts):
        result_file.write(str(classes[index]))
        line = [0] * len(classes)
        for class0 in dicts[class1]:
            inx = classes.index(class0 + 1)
            line[inx] = dicts[class1][class0]
        line = [' ' + str(ele) for ele in line]
        result_file.write(''.join(line) + '\n')
    result_file.close()


def visualize_dict(dict_root):
    f = open(dict_root, 'r')
    a = f.read()
    dicts = eval(a)
    result_file = open(dict_root.split('.txt')[0] +'_array.txt','w')
    # count mislabel prob
    mislabel = {}
    label = {}
    total = {}
    for class1 in dicts:
        mislabel_max = 0
        if class1 in dicts[class1]:
            label[class1] = dicts[class1][class1]
        else:
            label[class1] = 0
        num_total = sum(dicts[class1].values())
        total[class1] = num_total
        dicts[class1] = sorted(dicts[class1].items(), key=lambda d:d[1], reverse = True)
        for inx, ele in enumerate(dicts[class1]):
            if ele[0] != class1:
                mislabel_max = dicts[class1][inx][1]
                break
        mislabel[class1] = float(num_total - label[class1]) / num_total
    mislabel = sorted(mislabel.items(), key = lambda d:d[1], reverse = True)
    for tuple_class1 in mislabel:
        class1 = tuple_class1[0]
        dict_class1 = dicts[class1]
        line = [' ' + str(ele[0]) + ':' + str(ele[1]) for ele in dict_class1 if ele[0] != class1]
        #label total_num mislabel_ratio [mislabel result...]
        line_vis = str(class1) + ' ' + str(total[class1]) + ' ' + str(tuple_class1[1])
        line_vis = line_vis + ''.join(line)
        result_file.write(line_vis + '\n')
    result_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="locate model root")
    parser.add_argument('root', help='path to model root.')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    images_root = '/mnt/data-1/data/qi01.zhang/COCO/data/head/test0/'
    dict_root = '/mnt/data-1/data/qi01.zhang/COCO/model_data/' + args.root + '/matrix.txt'
    visualize_dict(dict_root)
