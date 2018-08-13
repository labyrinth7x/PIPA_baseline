base_root="/mnt/data-1/data/qi01.zhang/COCO/model_data"
python train_lfw.py --prefix "${base_root}/softmax-lr-0.1-wd-0.001-gamma-0.8-face-lfw/face" --epoch 200 --gpus 1 --region face --lr 0.1 --wd 0.001 --gamma 0.8 --train_root '/mnt/data-1/data/qi01.zhang/lfw/lst/train.rec'
