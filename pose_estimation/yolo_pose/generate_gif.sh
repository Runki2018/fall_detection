#!/bin/bash
python demos/test_my_video.py \
-p videos/1.mp4 \
--data data/mixed-data-kp.yaml \
--weights runs/mixed_v7tiny_e300_640/dm100_wm52_r904/weights/best.pt \
--device 0 \
--start 30 \
--end 300 \
--conf-thres 0.5 \
--iou-thres 0.45 \
--vis-thres 0.5 \
--kp-size 7 \
--line-thick 4 \
--imgsz 640 \
--alpha 0. \
--gif
#--no-kp-dets \

