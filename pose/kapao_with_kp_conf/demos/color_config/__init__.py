import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

def parse_color_config(dataset_name='coco'):
    if 'coco' in dataset_name:
        from coco import dataset_info
    elif 'mixed' in dataset_name:   # personal dataset: mixed-pose or pose12
        from mixedpose import dataset_info
    else:
        from crowdpose import dataset_info

    kpt_info = dataset_info['keypoint_info']
    skt_info = dataset_info['skeleton_info']
    name2index = {v['name']:k for k, v in kpt_info.items()}
    return kpt_info, skt_info, name2index


if __name__ == '__main__':
    _,_, a = parse_color_config()
    print(a)
