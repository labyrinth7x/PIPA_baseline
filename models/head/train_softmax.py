import argparse,logging,os
import mxnet as mx
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

def multi_factor_scheduler(factor, begin_epoch, epoch_size, step=[10,20]):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def get_network(args):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('./' + args.network, args.epoch_load)
    all_layers = symbol.get_internals()
    flatten0_output = all_layers['flatten0_output']
    softmax_label = mx.sym.Variable('softmax_label')
    net = mx.sym.Dropout(data = flatten0_output,p = 0.5)
    weight_net_init = mx.init.Xavier()
    #weight_512 = mx.sym.Variable('embed512_weight', init = weight_net_init, shape = (512,2048), lr = 3.0)
    #net = mx.FullyConnected(data = net, weight = weight_512, num_hidden = 512, name = 'embed512', no_bias = True)
    #net = mx.sym.Activation(data = net, act_type = 'relu', name = 'activation512')
    #net = mx.sym.Dropout(data = net, p = 0.5)
    weight_256 = mx.sym.Variable('embed_weight', init= weight_net_init, shape = (args.embed_size,2048),lr_mult = args.lr_mult)
    net = mx.sym.FullyConnected(data = net, weight = weight_256, num_hidden = args.embed_size, name = 'embed', no_bias = True)
    net = mx.sym.Activation(data = net, act_type = 'relu', name = 'activation256')
    #softmax branch 
    weight_softmax_init = mx.init.Normal(0.001)
    weight_softmax = mx.sym.Variable('fc1_weight', shape = (2356,args.embed_size), lr_mult = args.lr_mult, init = weight_softmax_init)
    net_softmax = mx.sym.FullyConnected(data = net, weight = weight_softmax, num_hidden = 2356, name = 'fc1', no_bias = True)
    net_softmax = mx.sym.SoftmaxOutput(data = net_softmax,name ='softmax_loss',label = softmax_label)
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return net_softmax, new_args, aux_params

def resume_network(args):
    symbol, arg_params, aux_params = mx.model.load_checkpoint('/mnt/data-1/data/qi01.zhang/COCO/models/head/softmax/head/no-pretrained-20epoch', 26)
    all_layers = symbol.get_internals()
    net_softmax = all_layers['softmax_loss_output']
    return net_softmax, arg_params, aux_params

def get_data_iter(args, kv):
    mean_r,mean_g,mean_b,scale = 0,0,0,1
    if args.aug_level >= 1:
        args.random_crop = 1
        args.mirror = 1
    if args.reid_crop == 1:
        args.random_crop = 0
    if args.aug_level >= 2:
        args.max_random_h = 36
        args.max_random_s = 50
        args.max_random_l = 50
    if args.aug_level >= 3:
        args.max_random_rotate_angle = 10
        args.max_random_shear_ratio = 0.1
        args.max_random_aspect_ratio = 0.25
    if args.region == 'head':
        shape = (3,224,224)
        resize = 224
        if args.reid_crop == 1:
            args.min_shorter_edge = 224
            args.max_shorter_edge = 300
        if args.mean_scale:
            mean_r = 66.427663666
            mean_g = 51.5901070721
            mean_b = 44.509282825
            scale = 0.0078125
    elif args.region == 'person':
        if args.person_padding:
            args.train_root = args.train_root.replace('head','person_padding')
            shape = (3,560,224)
        else:
            args.train_root = args.train_root.replace('head', 'person_anno')
            shape = (3,256,128)
        resize = -1
        if args.reid_crop == 1:
            args.min_shorter_edge = 128
            args.max_shorter_edge = 200
        if args.mean_scale:
            mean_r = 109.331541373
            mean_g = 94.3656742673
            mean_b = 88.0521883656
            scale = 0.0078125
    elif args.region == 'face':
        args.train_root = args.train_root.replace('head', 'face_anno')
        resize = -1
        shape = (3,112,96)
        if args.reid_crop == 1:
            args.min_shorter_edge = 96
            args.max_shorter_edge = 150
        if args.mean_scale:
            mean_r = 138.53685211
            mean_g = 100.649704666
            mean_b = 85.3696663188
            scale = 0.0078125
    train = mx.io.ImageRecordIter(
        path_imgrec = args.train_root,
        data_name = "data",
        label_name = "softmax_label",
        data_shape = shape,
        mirror = args.mirror,
        rand_crop = args.random_crop,
        reid_crop = args.reid_crop,
        min_shorter_edge = args.min_shorter_edge,
        max_shorter_edge = args.max_shorter_edge,
        random_h = args.max_random_h,
        random_s = args.max_random_s,
        random_l = args.max_random_l,
        max_rotate_angle = args.max_random_rotate_angle,
        max_shear_ratio = args.max_random_shear_ratio,
        min_random_scale = args.min_random_scale,
        max_random_scale = args.max_random_scale,
        max_aspect_ratio = args.max_random_aspect_ratio,
        batch_size = args.batch_size,
        resize = resize,
        round_batch = 0,
        num_parts = kv.num_workers,
        part_index = kv.rank,
        mean_r = mean_r,
        mean_g = mean_g,
        mean_b = mean_b,
        scale = scale
    )
    return train

 

def main(args):
    ##set parameters
    lr = args.lr
    wd = args.wd
    checkpoint = mx.callback.do_checkpoint(args.prefix)
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    if args.region == 'person':
        num_examples = 27112
    elif args.region == 'head':
        num_examples = 29221
    elif args.region == 'face':
        num_examples = 24734
    epoch_size = max(int(num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = 0
    #data, network
    train = get_data_iter(args,kv)
    network, arg_params, aux_params = get_network(args)
    mod = mx.mod.Module(symbol = network, context = devs)
    mod.bind(for_training = True,data_shapes = train.provide_data,label_shapes = train.provide_label)
    mod.fit(
        train_data = train,
        eval_data = None,
        aux_params = aux_params,
        arg_params = arg_params,
        eval_metric = ['acc','ce', mx.metric.create('top_k_accuracy', top_k = 5)],
        # eval_metric = [Accuracy(names = ['net_coco','net_softmax']),CrossEntropyLoss(names = ['net_coco','net_softmax'])],
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint,
        kvstore = kv,
        optimizer = 'sgd',
        #optimizer_params = {'learning_rate':args.lr,'wd':args.wd, 'momentum':0.9},
        optimizer_params = {'learning_rate':args.lr,'wd':args.wd, 'momentum':0.9, 'lr_scheduler':multi_factor_scheduler(args.gamma,begin_epoch,epoch_size, step = range(args.epoch_change,args.epoch,args.epoch_change))},
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2.34),
        num_epoch = args.epoch,
        begin_epoch = begin_epoch,
        allow_missing = True
    )

def add_data_aug_args(parser):                                                                                                          
    aug = parser.add_argument_group('Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=1,help='if or not randomly crop the image')
    aug.add_argument('--mirror', type=int, default=1,help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0, help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,help='max change of aspect ratio, whose range is [0, 1]')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    aug.add_argument('--min_shorter_edge', type = float, default = 0)
    aug.add_argument('--max_shorter_edge', type = float, default = 0)
    return aug          


def parse_args():
    parser = argparse.ArgumentParser(description="command for training resnet")
    parser.add_argument('--network', type = str, default = 'resnet-50')
    parser.add_argument('--epoch_load', type = int, default = 0) 
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--train_root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/head/lst/train.rec', help='train root')
    #parser.add_argument('--test0-root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/lst/test0.rec',help='test0 root')
    #parser.add_argument('--test1-root', type=str, default='/mnt/data-1/data/qi01.zhang/COCO/data/lst/test1.rec',help='test1 root')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--batch_size', type=int,default=64)
    parser.add_argument('--lr',type = float,default= 0.1)
    parser.add_argument('--wd',type = float,default = 0.001)
    parser.add_argument('--prefix',type=str)
    parser.add_argument('--gamma', type = float, default = 0.9)
    parser.add_argument('--epoch', type = int, default = 120)
    parser.add_argument('--epoch_change', type = int, default = 10)
    parser.add_argument('--embed_size', type = int, default = 256)
    parser.add_argument('--region', type = str, default = 'head')
    parser.add_argument('--lr_mult', type = float, default = 1.0)
    parser.add_argument('--aug_level', type = int, default = 0) 
    parser.add_argument('--mean_scale', action = 'store_false')
    parser.add_argument('--person_padding', action = 'store_true')
    parser.add_argument('--reid_crop', action = 'store_true')
    add_data_aug_args(parser)
    args = parser.parse_args()
    return args

def record(args):
    if args.reid_crop == 1:
        args.random_crop = 0
    folder = args.prefix[:args.prefix.rfind('/')]
    if not os.path.exists(folder):
        os.makedirs(folder)
    record = open(folder + '/config.txt', 'w')
    record.write('lr:%f'%args.lr+'\n')
    record.write('wd:%f'%args.wd+'\n')
    record.write('gamma:%f'%args.gamma+'\n')
    record.write('batch_size:%d'%args.batch_size+'\n')
    record.write('num_epoch:%d'%args.epoch+'\n')
    record.write('network:' + args.network + '\n')
    record.write('epoch_change:%d'%args.epoch_change + '\n')
    record.write('embed_size:%d'%args.embed_size + '\n')
    record.write('region:' + args.region + '\n')
    record.write('lr_mult:%f'%args.lr_mult + '\n')
    record.write('aug_level:%f'%args.aug_level + '\n')
    record.write('mean_scale:' + str(args.mean_scale) + '\n')
    record.write('person_padding:' + str(args.person_padding) + '\n')
    record.write('reid_crop:' + str(args.reid_crop))
    handler = logging.FileHandler(folder + '/train.log')                                                                                   
    handler.setLevel(logging.INFO)                                                                                                      
    formatter = logging.Formatter('%(message)s')                                                                                           
    handler.setFormatter(formatter)                                                                                                     
    logger.addHandler(handler) 


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    record(args)
    main(args)
