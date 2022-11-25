import cv2
import numpy as np 
from pathlib import Path
from get_video_pose import poseDetector
from pose_utils import pose_structure
import json
from tqdm import  tqdm

def _int(x, y):
    return int(x), int(y)

def generate_annotations(data_path):
    try:
        ann_path = Path(data_path).joinpath("Annotation_files")
        video_path = Path(data_path).joinpath("Videos")
        out_path = Path(data_path).joinpath("annotations")
        if not out_path.exists():
            out_path.mkdir(mode=0o777, parents=True, exist_ok=True)

        cv2.namedWindow("videos", cv2.WINDOW_KEEPRATIO)
        print("Press 'q' to exit this function.")
        txt_files = list(ann_path.glob("*.txt"))
        num_txt = len(txt_files)
        for ann_txt in tqdm(txt_files, desc=data_path, total=num_txt):
            with ann_txt.open('r') as fd:
                lines = fd.readlines()
                lines = [l.strip('\n') for l in lines if l.strip(' ') != '']

            fall_frame, frame_info = [], []
            for line in lines:
                if ',' not in line:
                    fall_frame.append(line)
                else:
                    frame_info.append(line.replace(',', ' '))

            fall_frame_start = int(fall_frame [0])  # 跌倒动作起始帧
            fall_frame_end = int(fall_frame [1])    # 跌倒动作结束帧， 跌倒动作后躺地上的帧数
            frame_info = np.loadtxt(frame_info)   

            video_file = video_path.joinpath(ann_txt.stem + ".avi")
            cap = cv2.VideoCapture(str(video_file))
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detector = poseDetector(frame_size=(height, width))
            annotations_list = []

            # fall_direction 是跌倒方向，顺时针从1-8, 1是正北(默认值), 2是东北, 3是正东, ....
            for frame_id, fall_direction, x1, y1, x2, y2 in frame_info:
                if cap.isOpened() and frame_id < frame_count:
                    _, img = cap.read()

                    if img is None:
                        continue

                    # detect pose 
                    img, results = detector.findPose(img, draw=False)
                    kpts, xyxy, area = detector.findPosition(img, results, pose_structure, draw=True)  # landmark

                    # draw txt info
                    img = cv2.rectangle(img, _int(x1, y1), _int(x2, y2), color=(0, 255, 0), thickness=1)
                    text = f"{frame_id:2.0f}, {fall_direction:2.0f}"
                    img = cv2.putText(img, text, (14, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color=(0, 255, 0), thickness=1)

                    if fall_frame_start <= frame_id <= fall_frame_end:
                        label = 1
                        img = cv2.putText(img, "falling", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color=(0, 0, 255), thickness=1)
                    elif fall_direction == 1:
                        label = 0
                        img = cv2.putText(img, "normal", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color=(0, 255, 0), thickness=1)
                    elif fall_direction != 1 or frame_id > fall_frame_end:
                        label = 2
                        img = cv2.putText(img, "faint", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color=(255, 255, 0), thickness=1)
                    
                    if xyxy != []:  # get more exact bbox
                        dx1, dy1, dx2, dy2 = [int(b) for b in xyxy]
                        x1, y1 = max(x1, dx1), max(y1, dy1)
                        x2, y2 = min(x2, dx2), min(y2, dy2)
                    
                    if not isinstance(kpts, list):  # not empty
                        kpts = kpts.flatten().tolist()

                    annotations_list.append(dict(
                        width=width,
                        height=height,
                        frame_id = frame_id,
                        bbox=[x1, y1, x2-x1, y2-y1],
                        keypoints=kpts,
                        fall_direction=fall_direction,   # 1~8
                        label=label   # normal 0 | falling 1 | faint 2
                    ))

                    cv2.imshow("videos", img)
                    key = cv2.waitKey(1)
                    if key & 0XFFFF == ord('q'):
                        print(f"quit!")
                        return

            cap.release()
            # save a json file per video
            json_dict = dict(
                info=dict(
                    label="normal 0 | falling 1 | faint 2",
                    fall_direction="int 1~8",
                    bbox="x1, y1, w, h",
                    keypoints="x1, y1, v1, ...., x12, y12, v12",
                    keypoints_order=['left_shoulder', 'right_shoulder',
                                    'left_elbow', 'right_elbow',        
                                    'left_wrist', 'right_wrist',       
                                    'left_hip', 'right_hip',         
                                    'left_knee', 'right_knee',        
                                    'left_ankle', 'right_ankle'],
                    skeleton = [[0, 1], [6, 7],
                                [0, 2], [1, 3],
                                [2, 4], [3, 5],
                                [0, 6], [1, 7],
                                [6, 8], [7, 9],
                                [8, 10], [9, 11]],  # 12个关键点的骨骼连线
                ),
                annotations=annotations_list
            )
            json_file = out_path.joinpath(ann_txt.stem + ".json")
            with json_file.open('w') as fp:
                json.dump(json_dict, fp, indent=4)

    except ValueError: 
        print(f"value error: ann_file => {ann_txt}")
        print("Check the txt file and assure that ")
        print("the frame number of the beginning of the fall")
        print("the frame number of the end of the fall 	")
        # 将标记跌倒起始帧和结束帧的数字放到最开始的两行。
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    train_data_roots = [
        # "FallDataset/Coffee_room_01",
        "FallDataset/Coffee_room_02",
        # "FallDataset/Home_01",
        # "FallDataset/Home_02",
    ]
    for data_root in train_data_roots:
        generate_annotations(data_root)