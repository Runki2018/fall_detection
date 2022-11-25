python val.py \
--data data/mixed-data-kp.yaml \
--weights runs/mixed_v7tiny_e300_640/kp_bbox10/weights/best.pt \
--batch-size 8 \
--imgsz 640 \
--task val \
--device 0 \
--conf-thres 0.01 \
--iou-thres 0.65 \
--conf-thres-kp 0.2 \
--conf-thres-kp-person 0.3 \
--iou-thres-kp 0.25 \
--overwrite-tol 50 \
#--no-kp-dets

