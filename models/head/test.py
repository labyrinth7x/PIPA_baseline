import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--test', action='store_false')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.test)
    if args.test:
        print('aaa')
