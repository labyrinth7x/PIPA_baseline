root="/mnt/data-7/data/qi01.zhang/COCO/model_data/"
for lr in {0.1,0.05,0.01,0.005};do
	for wd in {0.05,0.01,0.005,0.001,0.0005};do
		python train_softmax.py --gpus 1,2,3 --prefix "${root}/resent-152-softmax/head/lr-${lr}-wd-${wd}/head" --lr ${lr} --wd ${wd}
		#python test_softmax.py --gpus 1 --prefix "${root}/resnet-152-softmax/head/lr-${lr}-wd-${wd}/head" --lr ${lr} --wd ${wd} --out_file "${root}/resnet-152-softmax/head/out.txt"
	done
done

