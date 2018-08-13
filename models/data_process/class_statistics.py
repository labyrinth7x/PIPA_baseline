import os
from matplotlib import pyplot as plt

def save_class_statistics(base_root):
    if not os.path.exists(base_root + '/statistics'):
        os.mkdir(base_root + '/statistics')
    result_root = img_root + '_statistics.txt'
    dirs = {}
    result_file = open(result_root, 'w')
    folders = os.listdir(img_root)
    for folder in folders:
        num_imgs = len(os.listdir(img_root + '/' + folder))
        dirs[folder] = int(num_imgs)
        #result_file.write('class ' + folder + ' : %d'%num_imgs + '\n')
    dirs = sorted(dirs.items(),key = lambda x:x[1],reverse = True)
    result_file.write(str(dirs))

def draw_class_statistics(base_root, category):
    if not os.path.exists(base_root + '/statistics'):
        os.mkdir(base_root + '/statistics')
    folders = os.listdir(base_root + '/' + category)
    # dirs[#images] = #classes
    dirs = {}
    for folder in folders:
        num_imgs = len(os.listdir(base_root + '/' + category + '/' + folder))
        if num_imgs not in dirs:
            dirs[num_imgs] = 1
        else:
            dirs[num_imgs] += 1
    plt.hist(dirs.values(), bins = len(dirs))
    plt.xlabel('Num.Images per Class')
    plt.ylabel('Num.Class')
    plt.xlim((0))
    plt.title('PIPA ' + category + ' data distribution')
    plt.savefig(base_root + '/statistics/' + category + '.jpg')
    #plt.show()


if __name__ == "__main__":
    root = '/mnt/data-1/data/qi01.zhang/COCO/data/person_anno/data_processed'
    categories = ['train','val','test0','test1']
    for ca in categories:
        draw_class_statistics(root, ca)
