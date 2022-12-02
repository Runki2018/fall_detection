#!/bin/bash
python demos/test_my_video.py \
-p videos/1.mp4 \
--data data/mixed-data-kp.yaml \
--weights runs/mixed_v7tiny_e200_640/dm100_wm52_r90_fv2/weights/best.pt \
--device 0 \
--start 30 \
--end 300 \
--kp-thick -1 \
--line-thick -1 \
--conf-thres 0.5 \
--iou-thres 0.50 \
--conf-thres-kp 0.01 \
--conf-thres-kp-person 0.01 \
--iou-thres-kp 0.3 \
--overwrite-tol 70 \
--imgsz 640 \
--alpha 0. \
--no-kp-dets \
--gif

