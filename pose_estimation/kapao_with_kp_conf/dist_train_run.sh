# 1 add 12423 images from fall dataset
# 2 change the probability of mosaic to 0.85 (default to 1.0)
python my_train.py \
--img 512 \
--batch 96 \
--workers 9 \
--port 12345 \
--device 0,1,2 \
--epochs 60 \
--patience 20 \
--project runs/pose12_v7tiny_e100_512 \
--name fintune_mosaic001 \
--val-scales 1 \
--val-flips -1 \
--cfg models/yolov7-tiny-pose.yaml \
--data data/pose12-kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--sync-bn \
--multi-scale \
--weights "runs/pose12_v7tiny_e100_512/with_conf_mixup_cutpout_ms/weights/best.pt"
#--resume

#--search-cfg \
#--cfg models/yolov5s.yaml \
#--data data/coco-kp_10k.yaml \
#--hyp data/hyps/hyp.kp.yaml \
#--resume runs/s_e500_640/exp3/weight/last.pt
#--weights yolov5s.pt \
