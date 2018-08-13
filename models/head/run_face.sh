base_root="/mnt/data-1/data/qi01.zhang/COCO/model_data"
lr=0.2
wd=0.0
epoch=120
gpus=3
gamma=0.8
#for size in {256,128,64};do
size=256
python train_softmax_inception.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-mean-scale-aug-1-reid_crop-face-inception/face" --region face --lr ${lr} --wd ${wd} --gamma 0.6 --batch_size ${size} --lr_mult 1 --gpus ${gpus} --epoch ${epoch} --network Inception-BN --aug_level 1 --reid_crop
python test_softmax_epoch.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-mean-scale-aug-1-reid_crop-face-inception/face" --region face --gpus ${gpus} --epoch ${epoch} --epoch_begin 1 --interval 1 --restart
#done
