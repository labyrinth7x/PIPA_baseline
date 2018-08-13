from utils import point_distances
import random
import numpy as np

class KMeans:
    def __init__(self, k, feas):
        self.k = k
        # group_id stores the group_id of the feas
        # group_centroid stores the centeroid of the groups
        self.init_cluster(feas)


    # feas should be numpy
    def init_cluster(self, feas, method = True):
        self.group_centroid = []
        self.group_id = []
        if method is True:
            inx = random.randint(0, self.k-1)
            index_chosen = [inx]
            self.group_centroid.append(feas[inx])
            distances = point_distances(feas, feas)
            k = self.k
            while k > 1:
                distance = distances[inx]
                distance[inx] = float('-inf')
                index = np.argmax(distance)
                while index in index_chosen:
                    distance[index] = float('-inf')
                    index = np.argmax(distance)
                index_chosen.append(index)
                k -= 1
                self.group_centroid.append(feas[index])
            self.assign_group(feas)

    def update(self, feas):
        flag = self.k * [0]
        for index, fea in enumerate(feas):
            inx = self.group_id[index]
            if flag[inx] == 0:
                self.group_centroid[inx] = fea
            else:
                self.group_centroid[inx] += fea
            lag[inx] += 1
        k = self.k
        while k > 0:
            self.group_centroid[k - 1] = self.group_centroid[k - 1] / flag[k - 1]
            k -= 1
        self.assign_group(feas)

    def assign_group(self, feas):
        self.group_id = []
        for fea in feas:
            distance = point_distances(fea,self.group_centroid)
            self.group_id.append(np.argmin(distance))


    def reset(self, feas):
        init_cluster(self, feas)

    def get_group_centroid(self, index):
        group_id = self.group_id[index]
        group_centroid = self.group_centroid[group_id]
        return group_centroid


def visualize_kmeans():
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.network_location, args.epoch_load)
    net_embed = symbol.get_internals()['embed_output']
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    mod = mx.mod.Module(symbol = net, context=devs,label_names = None)
    mod.bind(for_training=False,data_shapes = test0.provide_data)
    mod.set_params(arg_params, aux_params)
    outputs = mod.predict()
    outputs = outputs.asnumpy()
    f = KMeans(args.groups, outputs)
    for batch in test0:
        label = batch.label[0]
        labels.append(label.asscalar())


def parse_args():
    parser = argparse.ArgumentParser(description='command for visualizing cluster')
    parser.add_argument('--network_location', type = str)
    parser.add_argument('--gpus', type = str, default = '0')
    parser.add_argument('--epoch_load', type = int)
    parser.add_argument('--kv_store', type = str, default ='device')
    parser.add_argument('--groups', type = int, default = 2356 * 10)
    args = parser.parse_args()
    return args


if __name__=="__main__":
    feas = np.array([[3,0],[0,1],[0,3]])
    f = KMeans(2,feas)


