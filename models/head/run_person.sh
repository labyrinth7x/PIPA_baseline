base_root="/mnt/data-1/data/qi01.zhang/COCO/model_data"
lr=0.2
aug=1
embed=256
gamma=0.7
wd=0
size=64
gpus=2
epoch=200
for size in {128,256};do
    python train_softmax_inception.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-aug-${aug}-reid_crop-change-20-person/body" --region person --lr ${lr} --wd ${wd} --gamma ${gamma} --batch_size ${size} --lr_mult 1 --gpus ${gpus} --epoch ${epoch} --epoch_change 20 --aug_level ${aug} --embed_size ${embed} --reid_crop
    python test_softmax_epoch.py --prefix "${base_root}/softmax-lr-${lr}-wd-${wd}-gamma-${gamma}-batch_size-${size}-aug-${aug}-reid_crop-change-20-person/body" --region person --gpus ${gpus} --epoch ${epoch} --epoch_begin 1 --interval 1 --restart
done
