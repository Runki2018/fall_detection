#python -m torch.distributed.launch --nproc_per_node 1 train.py \
python -m torch.distributed.run --nproc_per_node 1 train.py \
--img 640 \
--batch 32 \
--workers 8 \
 --device 2 \
--epochs 500 \
--data data/coco-kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5s.pt \
--project runs/s_e500_640 \
--resume runs/s_e500_640/exp3/weights/last.pt \
--name train
#--init_method tcp://127.0.0.1:45612 \
