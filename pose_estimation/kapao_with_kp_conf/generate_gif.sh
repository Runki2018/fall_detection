#!/bin/bash
python demos/test_my_video.py \
-p videos/1.mp4 \
--data data/pose12-kp.yaml \
--weights runs/pose12_v7tiny_e100_512/fintune_mosaic001/weights/last.pt \
--device 0 \
--start 0 \
--end -1 \
--kp-thick -1 \
--line-thick -1 \
--conf-thres 0.3 \
--iou-thres 0.50 \
--conf-thres-kp 0.1 \
--conf-thres-kp-person 0.2 \
--iou-thres-kp 0.5 \
--overwrite-tol 70 \
--imgsz 512 \
--alpha 0. \
--swap-nms \
--vis-kp-conf 0.1 \
--vis-conf-values \
--ow-ratios 0.1 0.1 0.15 0.15 0.2 0.2 0.1 0.1 0.2 0.2 0.3 0.3
#--no-kp-dets \
#--gif

