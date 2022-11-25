#CUDA_VISIBLE_DEVICES=1 WORLD_SIZE=1 \
#python -m torch.distributed.run --nproc_per_node 1 \
#--master_port 47770 \
python my_train.py \
--img 640 \
--batch 144 \
--workers 9 \
--port 12345 \
--device 0,1,2 \
--epochs 300 \
--patience 60 \
--project runs/mixed_v7tiny_e300_640 \
--name dm100_wm52_r90 \
--val-scales 1 \
--val-flips -1 \
--cfg models/yolov7-tiny-pose.yaml \
--data data/mixed-data-kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--weights runs/mixed_v7tiny_e300_640/dm100_wm52_r902/weights/best.pt
#--resume
#--sync-bn \
#--search-cfg \
#--cfg models/yolov5s.yaml \
#--data data/coco-kp_10k.yaml \
#--hyp data/hyps/hyp.kp.yaml \
#--resume runs/s_e500_640/exp3/weight/last.pt
#--weights yolov5s.pt \
