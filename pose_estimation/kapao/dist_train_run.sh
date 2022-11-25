# 1 add 12423 images from fall dataset
# 2 change the probability of mosaic to 0.85 (default to 1.0)
python my_train.py \
--img 640 \
--batch 144 \
--workers 9 \
--port 12345 \
--device 0,1,2 \
--epochs 300 \
--patience 60 \
--project runs/mixed_v7tiny_e300_640 \
--name kp_bbox10 \
--val-scales 1 \
--val-flips -1 \
--cfg models/yolov7-tiny-pose.yaml \
--data data/mixed-data-kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--sync-bn \
--weights runs/mixed_v7tiny_e200_640/dm100_wm52_r90_fv/weights/best.pt
#--resume
#--search-cfg \
#--cfg models/yolov5s.yaml \
#--data data/coco-kp_10k.yaml \
#--hyp data/hyps/hyp.kp.yaml \
#--resume runs/s_e500_640/exp3/weight/last.pt
#--weights yolov5s.pt \
