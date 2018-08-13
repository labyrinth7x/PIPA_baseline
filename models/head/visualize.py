import matplotlib.pyplot as plt
import os
import cv2


# visualize the image size distribution of the specific image set
def visualize_distribution(images_root):
    pointX = []
    pointY = []
    folders = os.listdir(images_root)
    for folder in folders:
        if not os.path.isdir(images_root + '/' + folder):
            continue
        imgs = os.listdir(images_root + '/' + folder)
        for img in imgs:
            height, width = cv2.imread(images_root + '/' + folder + '/' + img).shape[:2]
            pointX.append(width)
            pointY.append(height)
    plt.scatter(pointX, pointY, s = 30, c = 'red')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.savefig(images_root + '/face_train_size_distribution.jpg')
    plt.show()





def visualize_curve(log_root):
    log_file = open(log_root, "r")
    result_root = log_root[:log_root.rfind('/') + 1 ] + 'train.jpg'
    ac = []
    loss = []
    for line in log_file.readlines():
        line = line.strip().split()
        if len(line) < 8:
            continue
        if 'accuracy' in line[6]:
            string = line[6]
            ac.append(float(string.split('=')[1]))
        #if 'top_k_accuracy_5' in line[8]:
        #string = line[8]
        #ac_top5.append(float(string.split('=')[1]))
        if 'cross-entropy' in line[7]:
            string = line[7]
            loss.append(float(string.split('=')[1]))
    log_file.close()

    plt.figure('result')
    plt.subplot(211)
    plt.plot(ac)
    #plt.plot(ac_top5)
    plt.legend(["accuracy"],loc = 'upper right')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(loss)
    plt.legend(['cross_entropy'],loc = 'upper right')
    plt.grid(True)
    plt.savefig(result_root)
    #plt.show()

if __name__=='__main__':
    imgs_root = '/mnt/data-1/data/qi01.zhang/COCO/data/face_anno/data_unprocessed/train'
    #visualize_distribution(imgs_root)
    visualize_curve('/mnt/data-1/data/qi01.zhang/COCO/model_data/softmax-lr-0.1-wd-0.001-gamma-0.8-batch_size-64-face-mirror-inception/train.log')
