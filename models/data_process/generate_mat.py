


def generate_mat(index_files, result_root, key = 'head'):
    # seperate train, val
    dirs = []
    dir_train = []
    dir_val = []
    dir_test0 = []
    dir_test1 = []
    result_file = open(result_root + '/' + key + '.txt','w')
    train_val = open(index_files[0], 'r')
    lines = train_val.readlines()
    for line in lines:
        if not line:
            break
        dicts = line.strip().split()
        img_key = dicts[0] + '_' + dicts[1] + '.jpg'
        ca = int(dicts[-1])
        if ca == 1:
            dir_train.append(img_key)
        elif ca == 2:
            dir_val.append(img_key)
    dirs.append(dir_train)
    dirs.append(dir_val)
    test = open(index_files[1], 'r')
    lines = test.readlines()
    for line in lines:
        if not line:
            break
        dicts = line.strip().split()
        img_key = dicts[0] + '_' + dicts[1] + '.jpg'
        subset = int(dicts[-1])
        if subset == 0:
            dir_test0.append(img_key)
        else:
            dir_test1.append(img_key)
    dirs.append(dir_test0)
    dirs.append(dir_test1)
    result_file.write(str(dirs))


def load_mat(mat_root):
    mat_file = open(mat_root, 'r')
    f = mat_file.read()
    dirs = eval(f)

if __name__ == "__main__":
    base_root = '/mnt/data-1/data/qi01.zhang/'
    index_files = [base_root + 'PIPA/annotations/index.txt', base_root + 'COCO/data/data_split/test/split_test_original.txt']
    result_root = base_root + 'COCO/data/mat'
    # generate_mat(index_files, result_root, key = 'head')
    load_mat(result_root + '/head.txt')
