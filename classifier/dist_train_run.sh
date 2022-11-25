# 1 add 12423 images from fall dataset
# 2 change the probability of mosaic to 0.85 (default to 1.0)
python train.py \
--img 128 \
--batch 64 \
--workers 2 \
--port 12345 \
--device 2 \
--epochs 200 \
--patience 60 \
--project runs/cls_128 \
--name cls \
--hyp hyp_cls.yaml \
--weights "runs/cls_256/cls2/weights/best.pt"
#--adam \
#--sync-bn \
#--resume

#--search-cfg \
#--cfg models/yolov5s.yaml \
#--data data/coco-kp_10k.yaml \
#--hyp data/hyps/hyp.kp.yaml \
#--resume runs/s_e500_640/exp3/weight/last.pt
#--weights yolov5s.pt \
