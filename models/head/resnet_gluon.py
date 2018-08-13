import numpy as np
import mxnet
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import ndarray as nd
from mxnet.gluon.data.vision import transforms
import argparse

class L2norm(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(L2norm, self).__init__(**kwargs)
        
    def forward(self, x):
        return nd.L2Normalization(x, mode = 'instance')

def get_data(root, batch_size):
    data_iter = mxnet.io.ImageRecordIter(
            path_imgrec = root,
            data_shape = (3,224,224),
            batch_size = batch_size,
            resize = 224,
            round_batch = 0
            )
    return data_iter

def get_net(ctx, centroid = None):
    finetune_net = models.resnet50_v2(pretrained = True, ctx = ctx)
    # features and output seem to be seperated.
    finetune_net.output_new = nn.HybridSequential(prefix = '')
    # finetune_net.output_new.add(nn.InstanceNorm())
    finetune_net.output_new.add(L2norm())
    finetune_net.output_new.add(nn.Dense(2356))
    finetune_net.output_new.initialize(init.Xavier(),ctx = ctx)
    # finetune_net.output_new.add(nn.Dense(persons, activation = None, use_bias = False, weight_initializer = initializer.load(centroid)))
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net


def get_loss(data, net, ctx):
    loss = 0.0
    for feas, label in data:
        # epoch
        # return an array in the target device ctx with the same value as this array.
        label = label.as_in_context(ctx)
        # compute the output
        output_features = net.features(feas.as_in_context(ctx))
        # Every row holds an example
        # output_features_norm = [out / np.sqrt(np.sum(np.square(out))) for out in output_features]
        output = net.output_new(output_features)
        cross_entropy = nd.softmax_cross_entropy(output, label)
        loss += nd.mean(cross_entropy).asscalar()
    return loss / len(data)

def train(net, train_data, ctx, num_epochs = 1, lr_period = 10, lr_decay = 0.8, batch_size = 64):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':0.005,'wd': 0.005})
    for epoch in range(num_epochs):
        train_loss = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        index = 0
        for batch in train_data:
            index += 1
            label = batch.label[0] - 1
            data = batch.data[0]
            label = label.astype('float32').as_in_context(ctx)
            with autograd.record():
                output_features = net.features(data.as_in_context(ctx))
                # print(output_features[3])
                # for index in range(len(output_features)):
                   #  output_features[index] = output_features[index] / nd.sqrt(nd.sum((nd.square(output_features[index])))).asscalar()
                output = net.output_new(output_features)
                # for index in range(len(output)):
                    #output[index] = output[index] / nd.sqrt(nd.sum(nd.square(output[index])))
                # loss = softmax_cross_entropy(output, label)
                # print(label[3])
                loss = softmax_cross_entropy(output,label)
                # print(loss)
            loss.backward()
            trainer.step(batch_size, True)
            train_loss += nd.mean(loss).asscalar()
            # print(type(loss))
            # try:
                # loss.asnumpy()
            # except mxnet.base.MXNetError:
                # print("loss asnumpy fail")
                # continue
            print(nd.mean(loss).asscalar())
        epoch_str = ("Epoch %d. Train loss: %f, " % (epoch, train_loss / index))
        print(epoch_str + ', lr ' + str(trainer.learning_rate))
        train_data.reset()
    #net.hybridize()
    #net.export('resnet')

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Network running')
    parser.add_argument('train_root', help='train_root for train images.')
    parser.add_argument('test0_root', help='test0_root for test0 images.')
    parser.add_argument('test1_root', help='test1_root for test1 images.')
    # parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    args = parser.parse_args()
    return args

def test(net, test0_data, test1_data, ctx):
    labels = []
    accuracy = 0.0
    outputs = []
    for batch0 in test0_data:
        data = batch0.data
        label = batch0.label - 1
        output = net.features(data.as_in_context(ctx))
        outputs.extend(output.asnumpy())
        labels.extend(label)
    for batch1 in test1_data:
        label = batch1.label
        data = batch1.data
        out = net.features(data.as_in_context(ctx))
        sims = []
        for index in len(outputs):
            sim = np.dot(outputs[index],out) / np.linalg.norm(outputs[index])
            sims.append(sim)
        accuracy += (labels[np.argmax(sims)] == label)
    return accuracy


if __name__ == "__main__":
    batch_size = 64
    args = parse_args()
    train_data = get_data(args.train_root, batch_size = batch_size)
    test0_data = get_data(args.test0_root, batch_size = batch_size)
    test1_data = get_data(args.test1_root, batch_size = batch_size)
    ctx = mxnet.gpu(2)
    # ctx = args.gpus
    net = get_net(ctx)
    train(net, train_data, ctx, batch_size = batch_size)
    ac = test(net, test0_data, test1_data, ctx)
    print(ac)
