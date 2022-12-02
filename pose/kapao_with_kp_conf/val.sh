#!/bin/bash
#python val.py \
#--data data/pose12-kp.yaml \
#--weights runs/pose12_v7tiny_e100_512/fintune_mosaic001/weights/last.pt \
#--batch-size 8 \
#--imgsz 512 \
#--task val \
#--device 0 \
#--conf-thres 0.01 \
#--iou-thres 0.65 \
#--conf-thres-kp 0.2 \
#--conf-thres-kp-person 0.3 \
#--iou-thres-kp 0.25 \
#--overwrite-tol 50 \
#--swap_nms
#--no-kp-dets

weights_list=( \
"runs/pose12_v7tiny_e100_512/fintune_mosaic001/weights/last.pt" \
"runs/pose12_v7tiny_e100_512/with_conf/weights/best.pt" \
"runs/pose12_v7tiny_e100_512/with_conf2/weights/best.pt" \
"runs/pose12_v7tiny_e100_512/with_conf_mixup_cutpout_ms/weights/best.pt" \
)

for weights in ${weights_list[@]}
do
  echo $weights
  echo "without swap_nms"
  python val.py \
  --data data/pose12-kp.yaml \
  --weights $weights \
  --batch-size 8 \
  --imgsz 512 \
  --task val \
  --device 0 \
  --conf-thres 0.01 \
  --iou-thres 0.65 \
  --conf-thres-kp 0.2 \
  --conf-thres-kp-person 0.3 \
  --iou-thres-kp 0.25 \
  --overwrite-tol 50

  echo "with swap_nms"
  python val.py \
  --data data/pose12-kp.yaml \
  --weights $weights \
  --batch-size 8 \
  --imgsz 512 \
  --task val \
  --device 0 \
  --conf-thres 0.01 \
  --iou-thres 0.65 \
  --conf-thres-kp 0.2 \
  --conf-thres-kp-person 0.3 \
  --iou-thres-kp 0.25 \
  --overwrite-tol 50 \
  --swap_nms
done



