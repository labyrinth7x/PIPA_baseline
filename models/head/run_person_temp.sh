base_root="/mnt/data-1/data/qi01.zhang/COCO/model_data"
lr=0.005
aug=1
embed=256
gamma=0.8
wd=0
#size=64
gpus=1
epoch=120
for size in {32,64};do
    python train_caffenet.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-aug-${aug}-person-caffenet/body" --region person --lr ${lr} --wd ${wd} --gamma ${gamma} --batch_size ${size} --lr_mult 1 --gpus ${gpus} --epoch ${epoch} --aug_level ${aug} --embed_size ${embed} --network caffenet
    python test_softmax_epoch_caffenet.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-aug-${aug}-person-caffenet/body" --region person --gpus ${gpus} --epoch ${epoch} --epoch_begin 1 --interval 1 --restart
done
