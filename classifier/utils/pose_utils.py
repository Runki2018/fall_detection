# author: huangzhiyong
# date: 2022/10/19

import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import json
import mediapipe as mp 
import math
from pathlib import  Path
from tqdm import tqdm
import os
import shutil
from datetime import datetime as dt


# Medipipe 中检测的关键点结构
pose_structure = dict(
    keypoints=[
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',  # 11, 12
        'left_elbow', 'right_elbow',        # 13, 14
        'left_wrist', 'right_wrist',        # 15, 16
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
        'left_hip', 'right_hip',          # 23, 24
        'left_knee', 'right_knee',        # 25, 26
        'left_ankle', 'right_ankle',      # 27, 28
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ],
    request_kpt_indices = [11, 12,   # 挑选出所需的12个关键点序号
                            13, 14,
                            15, 16,
                            23, 24,
                            25, 26,
                            27, 28],
    skeleton = [[0, 1], [6, 7],
                [0, 2], [1, 3],
                [2, 4], [3, 5],
                [0, 6], [1, 7],
                [6, 8], [7, 9],
                [8, 10], [9, 11]],  # 12个关键点的骨骼连线
    line_color = [(255, 255, 255), (255, 255, 255),
                  (255, 0, 0),(255, 0, 0),
                  (0, 255, 0), (0, 255, 0),
                  (0, 0, 255), (0, 0, 255),
                  (255, 255, 0), (255, 255, 0),
                  (255, 0, 255), (255, 0, 255)]  # 12个关键点的骨骼连线的颜色
)

class poseDetector():
    """
        可视化视频、并保存可视化后的每一帧，用于后续处理。
        关键点蓝色点表示置信度高, 粉色点表示置信度低。
        目前仅仅适合单人视频检测, 可以获取关键点和边界框, 通过修改也可以获取图像分割掩码信息。
    """
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detection_confidence=0.5, track_confidence=0.5, frame_size=(640, 640)):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth   # True 减少抖动
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # MediaPipe Pose检测器参数,关闭防抖动,因为防抖动可能会降低动作大幅度变化时的检测精度
        self.pose = self.mpPose.Pose(static_image_mode=False, # True：图片模式, false 视频流模式
                                    model_complexity=2,       # 0, 1, 2, 越大精度越高,延迟越大
                                    smooth_landmarks=False,    # 减少关键点抖动,只对视频流有效
                                    enable_segmentation=True,  # 生成图像分割掩码, 这里用于获取人体边界框
                                    smooth_segmentation=False,   # 减少分割掩码抖动,只对视频流有效
                                    min_detection_confidence=0.5,  # 最小检测阈值, 预测结果大于该值保留
                                    min_tracking_confidence=0.5)   # 最小跟踪阈值, 检测结果和跟踪预测结果相似度
        self.grid = np.arange(frame_size[0]*frame_size[1]).reshape(frame_size)  # 网格序号,用于找出mask的区域,从而计算bbox
        self.mask_thr = 0.2   # 分割得分阈值,大于该值为人体像素
        self.thickness = 1
        self.text_position = (10, 10)

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, 
                results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS)
        return img, results

    def findPosition(self, img, results, pose_structure, draw=True):
        kpts, xyxy, area = [], [], 0
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                vis = round(lm.visibility, 4)
                # print(f"{cx=}\t{cy=}\t{vis=}")
                kpts.append([cx, cy, vis])
                if draw and idx in pose_structure['request_kpt_indices']:
                    color = (255, 0, 0) if vis > 0.8 else (128, 12, 255)
                    cv2.circle(img, (cx, cy), self.thickness+2, color, cv2.FILLED)

            kpts = np.array(kpts)[pose_structure['request_kpt_indices']]
            if draw:
                for pair, color in zip(pose_structure['skeleton'], pose_structure['line_color']):
                    # line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
                    pt1 = int(kpts[pair[0], 0]), int(kpts[pair[0], 1])
                    pt2 = int(kpts[pair[1], 0]), int(kpts[pair[1], 1])
                    cv2.line(img, pt1, pt2, color, thickness=self.thickness)

            area = (results.segmentation_mask > self.mask_thr).size  # 人体分割图面积
            if area > 0:
                # print(results.segmentation_mask.shape)   # h, w
                assert results.segmentation_mask.shape == self.grid.shape, f"{results.segmentation_mask.shape} != {self.grid.shape}"
                mask = self.grid[results.segmentation_mask > self.mask_thr]   # mask_thr 是正样本的阈值, 大于该值为人体区域, 小于则为背景区域
                x = mask % self.grid.shape[1]  
                y = mask // self.grid.shape[1]
                xy = np.concatenate([x[:, None], y[:, None]], axis=-1)
                left_top = xy.min(axis=0)
                right_bottom = xy.max(axis=0)
                xyxy = [*left_top, *right_bottom]
                cv2.rectangle(img, left_top, right_bottom, (255, 0, 0), thickness=self.thickness)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
                
        return kpts, xyxy, area


    def findAngle(self, img, kpts, p1, p2, p3, draw=True):
        x1, y1 = kpts[p1][1:]
        x2, y2 = kpts[p2][1:]
        x3, y3 = kpts[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle += 360 if angle < 0 else 0
        assert  0 <= angle <= 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    
        return angle        


def save_annotations(filename, kpts, bbox, height, width, area):
    json_dict = dict(
        height=height,
        width=width,
        area=area,
        keypoints=kpts.flatten().tolist(),
        bbox=bbox   # xywh
    )
    with open(str(filename), 'w') as fd:
        json.dump(json_dict, fd, indent=4)

def main(video_path:str,                            # 视频路径
         pose_dict,                                 # 需要保留和可视化的关键点结构
         interval=1,                                # 取帧间隔
         save=False,                                # 是否保存图片和标注
         save_img_root='images',                    # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式        
         save_ann_root='annotations',               # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
         name_with_video_path=False,                # 保存的图片名包括原视频所在的目录名
         show=True):                                # 是否显示可视化
    try:
        cap = cv2.VideoCapture(video_path)
        if show:
            cv2.namedWindow(video_path, cv2.WINDOW_KEEPRATIO)

        # 检查和创建输出目录
        if save:
            if name_with_video_path:
                img_path = Path(save_img_root).joinpath(Path(video_path).stem)
            else:
                img_path = Path(save_img_root).joinpath(Path(video_path).stem)
            if not img_path.exists():
                img_path.mkdir(mode=777, parents=True, exist_ok=True)
            
            ann_path = Path(save_ann_root).joinpath(Path(video_path).stem)
            if not ann_path.exists():
                ann_path.mkdir(mode=777, parents=True, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{video_path=}\n{width=}\n{height=}\n{frames_count=}")
        detector = poseDetector(frame_size=(height, width))

        for frame_idx in tqdm(range(frames_count), desc=video_path):
            if cap.isOpened():
                _, img = cap.read()
                if frame_idx % interval != 0:
                    continue
                img, results = detector.findPose(img, draw=False)
                kpts, xyxy, area = detector.findPosition(img, results, pose_structure=pose_dict, draw=True)  # landmark
                
                # add info on image
                pos = (30, 30)  # 文字框的左上角坐标
                cv2.putText(img, str(frame_idx), pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                if xyxy != []:
                    bbox = [int(b) for b in xyxy]
                    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]   # x1y1x2y2 -> x1y1wh
                    # if bbox[2]/bbox[3] > 1.2:   # 宽大于高度, 认为是跌倒
                    #     cv2.putText(img, "Fall", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    # else:
                    #     cv2.putText(img, "Normal", (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    
                
                if save and not isinstance(kpts, list):
                    img_file = img_path.joinpath(str(frame_idx) + '.jpg')
                    cv2.imwrite(str(img_file), img)
                    ann_file = ann_path.joinpath(str(frame_idx) + '.json')
                    save_annotations(ann_file, kpts, bbox, height, width, area)

                if show:
                    cv2.imshow(video_path, img)
                    cv2.waitKey(1)

    finally:
        cv2.destroyAllWindows()
        cap.release()
    
def vis_annotations(img, bbox, kpts):
    for i in range(0, len(kpts), 3):
        cv2.circle(img, (int(kpts[i]), int(kpts[i+1])), 6, (0, 255, 0), cv2.FILLED)
    left_top = bbox[:2]
    right_bottom = bbox[0] + bbox[2], bbox[1] + bbox[3]
    # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=2)  
    return img


def check_keypoints(kpts, width, height, thr=0.2):
    kpts = np.array(kpts).reshape((-1, 3))
    kpts[(kpts[:, 0] > width) & (kpts[:, 0] < 0)] = [0, 0, 0]  # 去除超出边界的关键点
    kpts[(kpts[:, 1] > height) & (kpts[:, 1] < 0)] = [0, 0, 0]  # 去除超出边界的关键点
    kpts[kpts[:, 2] < 0.2] = [0, 0, 0]  # 得分低于阈值的去除
    num_keypoints = np.sum(kpts[:, 2] > 0)
    return kpts.flatten().tolist(), num_keypoints


def merge_remained_data(video_root='videos',              # 存放视频的根目录
                        img_root='images',                # 存放高质量可视化图片的根目录,先使用window图片查看器去除低质量检测图片
                        ann_root='annotations',           # 存放标注文件的更目录
                        output_root='merge_datasets',     # 输出图片和标注文件的根目录。根据img_root中保留的图片,生成相应帧图片和标注文件
                        show=False,                       # 可视化帧与标注,查看是否正确匹配
                        ):
    """
    生成最终数据集
    1. 使用Windows图片查看器去除检测结果质量低的图片,快捷键: '→'键 下一张, 'Del'键 删除当前图片
    2. 使用merge_remained_data函数,根据保留下来的高质量检测结果,生成相应帧图片和标注文件
    """
    # 输入目录
    video_root = Path(video_root)
    img_dirs = os.listdir(img_root)   # 图片目录中我们手工删除检测质量较差的结果
    ann_dirs = os.listdir(ann_root)
    img_dirs = [d for d in img_dirs if os.path.isdir(os.path.join(img_root, d))]
    ann_dirs = [d for d in ann_dirs if os.path.isdir(os.path.join(img_root, d))]
    print(f"{img_dirs=}\t{ann_dirs=}")
    dirs = list(set(img_dirs) and set(ann_dirs))  # 共同视频目录
    video_names = {name.split('.')[0]:name for name in os.listdir(video_root) if name.split('.')[0] in dirs}
    print(f"common directories: {dirs}")

    # 输出目录
    output_root = Path(output_root)
    out_img_path = output_root.joinpath('images')
    out_ann_path = output_root.joinpath('annotations')
    if not out_img_path.exists():
        out_img_path.mkdir(mode=777, parents=True, exist_ok=True)
    if not out_ann_path.exists():
        out_ann_path.mkdir(mode=777, parents=True, exist_ok=True)

    print(f"Save images\t\t=> {out_img_path}")
    print(f"Save annotations\t=> {out_ann_path}")

    # 开始比对,如果合并保留下来的高质量检测结果。
    img_id, ann_id = 0, 0
    images, annotations = [], []   # Json标注文件中的
    num_videos = len(dirs)
    for idx, _dir in enumerate(dirs):
        try:
            img_files = os.listdir(os.path.join(img_root, _dir))
            ann_files = os.listdir(os.path.join(ann_root, _dir))
            hash_table = {a.strip('.json'):a for a in ann_files if '.json' in a}
            cap = cv2.VideoCapture(str(video_root.joinpath(video_names[_dir])))

            # 按帧序号升序排序
            img_files = sorted(img_files, key=lambda x:int(x.strip('.jpg')))
            
            # 匹配和生成标注
            for file in tqdm(img_files, desc=f"{idx}/{num_videos}: {_dir}"):
                frame_id = file.strip('.jpg')
                ann_file = hash_table.get(frame_id, None)
                if ann_file != None:
                    with open(os.path.join(ann_root, _dir, ann_file), 'r') as fd:
                        ann_dict = json.load(fd)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                    success, img = cap.read()
                    if not success:
                        continue
                    
                    dst_file = str(img_id) + '.jpg'
                    dst_img = out_img_path / dst_file
                    cv2.imwrite(str(dst_img), img)
                    
                    kpts, num_keypoints = check_keypoints(ann_dict['keypoints'], ann_dict['width'], ann_dict['height'], thr=0.2)
                    images.append(dict(
                        id=img_id,
                        file_name=dst_file,
                        width=ann_dict['width'],
                        height=ann_dict['height'],
                    ))
                    annotations.append(dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=1,
                        bbox=ann_dict['bbox'],
                        area=int(ann_dict['area']),
                        num_keypoints=int(num_keypoints),
                        keypoints=kpts,
                        iscrowd=0,
                    ))
                    img_id += 1
                    ann_id += 1
                    
                    if show:
                        vis_annotations(img, ann_dict['bbox'], ann_dict['keypoints'])
                        cv2.imshow('image', img)
                        cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            cap.release()

    # 保存COCO数据集标注格式的JSON标注文件
    json_file = out_ann_path.joinpath("video_keypoints.json")
    print(f"Saving annotation file\t=> {json_file}")
    with json_file.open('w') as fd:
        json_dict = dict(
            categories=[{
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder", "right_shoulder",
                    "left_elbow","right_elbow",
                    "left_wrist","right_wrist",
                    "left_hip","right_hip",
                    "left_knee","right_knee",
                    "left_ankle","right_ankle"
                ],
                "skeleton": [[0, 1], [6, 7],
                            [0, 2], [1, 3],
                            [2, 4], [3, 5],
                            [0, 6], [1, 7],
                            [6, 8], [7, 9],
                            [8, 10], [9, 11]]
            }],
            images=images,
            annotations=annotations)
        json.dump(json_dict, fd, indent=4)
    print(f"the number of images is {len(images)}")
    print(f"Done!")


def extract_videos(videos_root, output_dir):
    """将指定根目录下面的所有视频重命名(确保命名不冲突), 并输出到指定目录
    videos_root (str)
    output_dir (str)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(mode=777, parents=True, exist_ok=True)
    print(f"Extracting videos from {videos_root} to {str(output_dir)}")

    # 多个视频的根目录
    video_path = Path(videos_root)
    # 视频的格式如： '.avi', '.mp4'
    video_format = ['mov', 'avi', 'mp4','mpg','mpeg','m4v','mkv','wmv']
    print(f"Supported Video Formats => {video_format}")
    for vf in video_format:
        for video in video_path.glob('**/*' + vf):
            src_video = str(video)
            dst_video = dt.strftime(dt.now(), "%Y%m%d%H%M%S") + video.name
            dst_video = str(output_dir.joinpath(dst_video))
            shutil.copyfile(src_video, dst_video)
            print(f"src_video => dst_video: {src_video} => {dst_video}")


def pose_similarity(kp1, kp2, bbox1, bbox2, iou_thres=0.6, pck_thres=0.01):
    """计算当前帧与前一帧图片的姿态相似度, 如果过于相似, 则丢弃当前帧"""
    x1, y1, x2, y2 = bbox1
    xx1, yy1, xx2, yy2 = bbox2
    ix1, iy1 = max(x1, xx1), max(y1, yy1)
    ix2, iy2 = min(x2, xx2), min(y2, yy2)
    intersection = max(0, (ix2 - ix1)*(iy2 - iy1))
    w1, h1 = (x2-x1), (y2-y1)
    w2, h2 = (xx2-xx1), (yy2-yy1)
    union =  w1*h1  + w2*h2 - intersection
    iou = intersection / union
    if iou > iou_thres:
        scale_factor = (w1 + h1 + w2 + h2) / 4
        kp1 = np.array(kp1).reshape((-1, 3))[:, :2]
        kp2 = np.array(kp2).reshape((-1, 3))[:, :2]
        dist = np.linalg.norm((kp1-kp2), axis=1) / scale_factor
        pck = sum(dist < pck_thres) / kp1.shape[0]
    else:
        pck = 0
    return round(iou, 3), round(pck, 3)
