# fall_detection

## 1. Install Environment
```shell
pip install - r requirements.txt
cd object_track/botsort/
python setup.py develop
# Cython-bbox
pip3 install cython_bbox
# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

## 2. Train classifier
```shell
cd classifier
# you can customize the training configures on dist_train_run.sh
bash dist_train_run.sh
```