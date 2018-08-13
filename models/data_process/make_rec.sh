#for ca in {"train"};do
ca="train"
region="person"
base_root="/mnt/data-1/data/qi01.zhang/COCO/data/${region}_anno"
#python image_preprocess.py "${base_root}/data_unprocessed/${ca}" "${base_root}/data_processed/${ca}" ${region}
python makelist.py "${base_root}/lst_origin/" "${base_root}/data_unprocessed/${ca}"
python im2rec.py "${base_root}/lst_origin/${ca}.lst" "${base_root}/data_unprocessed/${ca}"
