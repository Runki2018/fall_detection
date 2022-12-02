from pycocotools.coco import COCO
import json
from pathlib import Path
from pycocotools.coco import COCO

kp_mpii={
        0:
        dict(
            name='right_ankle',
            id=0,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        1:
        dict(
            name='right_knee',
            id=1,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        2:
        dict(
            name='right_hip',
            id=2,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        3:
        dict(
            name='left_hip',
            id=3,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        4:
        dict(
            name='left_knee',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        5:
        dict(
            name='left_ankle',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        6:
        dict(name='pelvis', id=6, color=[51, 153, 255], type='lower', swap=''),
        7:
        dict(name='thorax', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(
            name='upper_neck',
            id=8,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        9:
        dict(
            name='head_top', id=9, color=[51, 153, 255], type='upper',
            swap=''),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='right_elbow',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        12:
        dict(
            name='right_shoulder',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        13:
        dict(
            name='left_shoulder',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        14:
        dict(
            name='left_elbow',
            id=14,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        15:
        dict(
            name='left_wrist',
            id=15,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist')
    }
kp_coco ={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    }
kp_crowdpose={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(
            name='top_head', id=12, color=[255, 128, 0], type='upper',
            swap=''),
        13:
        dict(name='neck', id=13, color=[0, 255, 0], type='upper', swap='')
    }

def read_json(path):
    with open(path, 'r') as fd:
        json_dict = json.load(fd)
    return json_dict

def get_part_data(json_file, num_samples=64):
    coco = COCO(annotation_file=json_file)
    img_ids = coco.getImgIds(catIds=1)[:num_samples]
    ann_ids = coco.getAnnIds(imgIds=img_ids)

    imgs = coco.loadImgs(img_ids)
    anns = coco.loadAnns(ann_ids)

    json_file = Path(json_file)

    json_dict = json.load(json_file.open(mode='r'))

    json_dict["images"] = imgs
    json_dict["annotations"] = anns

    new_json_file = json_file.with_name(json_file.stem + "_" + str(num_samples) + '.json')
    print(f"{json_file=} => \n{new_json_file=}")
    json.dump(json_dict, new_json_file.open(mode='w'), indent=4)


def merge_datasets(coco, mpii, crowdpose):
    t_coco = COCO(coco)
    t_crowd = COCO(crowdpose)

    coco = read_json(coco)
    mpii = read_json(mpii)
    crowdpose = read_json(crowdpose)

    print(f"{coco.keys()}")
    print(f"{mpii[0].keys()}")
    print(f"{crowdpose.keys()}")

    img_id, ann_id = 0, 0
    images_list = []
    annotation_list =[]
    coco_imgIds = t_coco.getImgIds()
    # for i in coco_imgIds:
    #      annIds = t_coco.getAnnIds(imgIds=i)
        # for j







if __name__ == '__main__':
    # get_part_data(json_file="./data/coco/annotations/person_keypoints_train2017.json",
    #               num_samples=10000)
    merge_datasets(
        coco="data/data/coco/annotations/person_keypoints_train2017.json",
        mpii="data/data/mpii/annotations/mpii_trainval.json",
        crowdpose="data/data/crowdpose/annotations/mmpose_crowdpose_trainval.json"
    )
