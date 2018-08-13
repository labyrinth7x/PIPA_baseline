base_root="/mnt/data-1/data/qi01.zhang/COCO/model_data"
aug=1
lr=5
wd=0.0
gamma=0.5
size=128
epoch=120
gpus=3
python train_softmax_inception.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-mean-scale-aug-${aug}/head" --region head --lr ${lr} --wd ${wd} --gamma ${gamma} --batch_size ${size} --lr_mult 1 --gpus ${gpus} --epoch ${epoch} --aug_level ${aug}
#python test_softmax_epoch.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-mean-scale-aug-${aug}/head" --region head --gpus ${gpus} --epoch ${epoch} --epoch_begin 1 --interval 1 --restart
