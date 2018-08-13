import random 
import argparse
import os


def list_image(root, exts):
    i = 0
    for index, folder in enumerate(sorted(os.listdir(root))):
        folder_root = root + "/" + folder
        for fname in sorted(os.listdir(folder_root)):
            if fname[0] == ".":
                continue
            fpath = os.path.join(folder_root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), int(index))
                i += 1


def make_list(args):
    image_list = list_image(args.root, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)
    write_list(args.prefix + "/" + args.root.split("/")[-1] +".lst", image_list)


def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            if i == len(image_list) - 1:
                fout.write(line.strip())
            else:
                fout.write(line)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst files.')
    parser.add_argument('root', help='path to folder containing images.')
    #parser.add_argument('labels', help='path to store labels.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args



if __name__ == "__main__":
    args = parse_args()
    make_list(args)
