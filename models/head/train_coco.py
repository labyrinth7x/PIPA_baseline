import argparse,logging,os
import mxnet as mx
import pudb
from mxnet import symbol as sym


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Accuracy(mx.metric.EvalMetric):
    def __init__(self, names = ['accuracy']):
        self.names = ["accuracy: " + name for name in names]
        self.sum_metric_list = [0.0] * len(self.names)
        self.num_inst_list = [0.0] * len(self.names)
        super(Accuracy, self).__init__('accuracy')
 
    def update(self, labels, preds):
        print("aaa")
        labels = labels[0].asnumpy().astype('int32')
        for index in range(len(self.names)):
            preds_labels = preds[index]
            preds_labels = mx.nd.argmax_channel(preds_labels)
            preds_labels = preds_labels.asnumpy().astype('int32')
            self.sum_metric_list[index] += ((preds_labels.flat == labels.flat).sum())
            self.num_inst_list[index] += len(preds_labels.flat)

    def reset(self):
        self.sum_metric_list =[0.0] * len(self.names)
        self.num_inst_list = [0.0] * len(self.names)

    def get(self):
        return (self.names, [self.sum_metric_list[index] / self.num_inst_list[index] for index in range(len(self.names))])


class CrossEntropyLoss(mx.metric.EvalMetric):
    def __init__(self, names = ['ce']):
        self.names = ["ce: " + name for name in names]
        self.ce_list = [0.0] * len(self.names)
        super(CrossEntropyLoss,self).__init__('ce')
 
    def update(self,labels, preds):
        print("Bbbb")
        labels = labels[0]
        for index in range(len(self.names)):
            self.ce_list[index] = 0.0
            preds_labels = preds[index]
            ce = mx.metric.CrossEntropy()
            ce.update(labels, preds_labels)
            self.ce_list[index] += ce.get()[1]
    

    def reset(self):
        self.ce_list = [0.0] * len(self.names)

    def get(self):
        return (self.names, self.ce_list)

def multi_factor_scheduler(begin_epoch, epoch_size, step=[10,20], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def get_network(args):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./resnet-50',0)
    all_layers = symbol.get_internals()
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    flatten0_output = all_layers['flatten0_output']
    softmax_label = mx.sym.Variable('softmax_label')
    net = mx.sym.Dropout(data = flatten0_output,p = 0.5)
    # weight_net_init = mx.init.Xavier()
    weight_net = mx.sym.Variable('embed_weight', shape = (256,2048), lr_mult = 1.0)
    net = mx.sym.FullyConnected(data = net, weight = weight_net, num_hidden = 256, name = 'embed')
    #softmax branch
    weight_softmax_init = mx.init.Normal(0.001)
    weight_softmax = mx.sym.Variable('fc1_weight', shape = (2356,256), lr_mult = 1.0, init = weight_softmax_init)
    net_softmax = mx.sym.FullyConnected(data = net, weight = weight_softmax, num_hidden = 2356, name = 'fc1')
    net_softmax = mx.sym.SoftmaxOutput(data = net_softmax,name ='softmax_loss',label = softmax_label, grad_scale = 1)
    #coco loss branch
    weight_coco = mx.sym.Variable('fc2_weight',shape = (2356,256), lr_mult= 1.0)
    weight_coco = mx.sym.L2Normalization(data = weight_coco, mode ='instance')
    net_coco = mx.sym.FullyConnected(data = net, weight = weight_coco, num_hidden = 2356, name = 'fc2',no_bias = True)
    net_coco = mx.sym.SoftmaxOutput(data = net_coco, name = 'coco_loss', label = softmax_label, grad_scale = 0.4)
    group = mx.sym.Group([net_coco,net_softmax])
    return group


def get_data(args,kv):
    train = mx.image.ImageIter(
        path_imgrec = args.train_root,
        data_name = "data",
        label_name = "softmax_label",
        data_shape = (3,224,224),
        batch_size = args.batch_size,
        resize = 224,
        num_parts = kv.num_workers,
        part_index = kv.rank
    )
    return train

def main(args):
    ##set parameters
    lr = args.lr
    wd = args.wd
    dir = args.prefix.split('/')
    if not os.path.exists('./'+dir[1]):
        os.mkdir('./'+dir[1])
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = 0
    #data, network
    train = get_data(args,kv)
    network = get_network(args)
    mod = mx.mod.Module(symbol = network, context = devs)
    mod.bind(for_training = True,data_shapes = train.provide_data,label_shapes = train.provide_label)
    mod.fit(
        train_data = train,
        eval_data = None,
        eval_metric = [Accuracy(names = ['net_coco','net_softmax']),CrossEntropyLoss(names = ['net_coco','net_softmax'])],
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint,
        kvstore = kv,
        optimizer = 'sgd',
        optimizer_params = {'learning_rate':lr,'wd':wd},
        # optimizer_params = {'learning_rate':0.1,'wd':0.001,'lr_scheduler':multi_factor_scheduler(0,30)},
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        num_epoch = 30,
        begin_epoch = 0
        #allow_missing = True
    )

def parse_args():
    parser = argparse.ArgumentParser(description="command for training resnet-50")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--train-root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/head/lst/train.rec', help='train root')
    parser.add_argument('--test0-root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/head/lst/test0.rec',help='test0 root')
    parser.add_argument('--test1-root', type=str, default='/data-sdb/qi01.zhang/COCO/data/lst/test1.rec',help='test1 root')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--num_examples',type=int,default=29220)
    parser.add_argument('--batch_size', type=int,default=10)
    parser.add_argument('--lr',type = float,default= 5)
    parser.add_argument('--wd',type = float,default = 0.005)
    parser.add_argument('--prefix',type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    main(args)
