# 1 add 12423 images from fall dataset
# 2 change the probability of mosaic to 0.85 (default to 1.0)
python train.py \
--img 128 \
--batch 64 \
--workers 4 \
--port 12345 \
--device 2 \
--epochs 200 \
--patience 60 \
--project runs/cls_128 \
--name cls_img \
--hyp hyp_cls.yaml \
--weights ""
#--adam \
#--weights "runs/cls_128/cls/weights/best.pt"
#--sync-bn \
#--resume

