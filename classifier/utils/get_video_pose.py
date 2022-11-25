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
from argparse import ArgumentParser
from pose_utils import extract_videos, pose_similarity, pose_structure


# Medipipe 中检测的关键点结构
# pose_structure = dict(
#     keypoints=[
#         'nose',
#         'left_eye_inner', 'left_eye', 'left_eye_outer',
#         'right_eye_inner', 'right_eye', 'right_eye_outer',
#         'left_ear', 'right_ear',
#         'mouth_left', 'mouth_right',
#         'left_shoulder', 'right_shoulder',  # 11, 12
#         'left_elbow', 'right_elbow',        # 13, 14
#         'left_wrist', 'right_wrist',        # 15, 16
#         'left_pinky', 'right_pinky',
#         'left_index', 'right_index',
#         'left_thumb', 'right_thumb',
#         'left_hip', 'right_hip',          # 23, 24
#         'left_knee', 'right_knee',        # 25, 26
#         'left_ankle', 'right_ankle',      # 27, 28
#         'left_heel', 'right_heel',
#         'left_foot_index', 'right_foot_index'
#     ],
#     request_kpt_indices = [11, 12,   # 挑选出所需的12个关键点序号
#                             13, 14,
#                             15, 16,
#                             23, 24,
#                             25, 26,
#                             27, 28],
#     skeleton = [[0, 1], [6, 7],
#                 [0, 2], [1, 3],
#                 [2, 4], [3, 5],
#                 [0, 6], [1, 7],
#                 [6, 8], [7, 9],
#                 [8, 10], [9, 11]],  # 12个关键点的骨骼连线
#     line_color = [(255, 255, 255), (255, 255, 255),
#                   (255, 0, 0),(255, 0, 0),
#                   (0, 255, 0), (0, 255, 0),
#                   (0, 0, 255), (0, 0, 255),
#                   (255, 255, 0), (255, 255, 0),
#                   (255, 0, 255), (255, 0, 255)]  # 12个关键点的骨骼连线的颜色
# )

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
                                    model_complexity=2,       # 0, 1, 2, 越大精度越高,延迟越大, 注意设置为0或2时需要下载模型
                                    smooth_landmarks=False,    # 减少关键点抖动,只对视频流有效
                                    enable_segmentation=True,  # 生成图像分割掩码, 这里用于获取人体边界框
                                    smooth_segmentation=False,   # 减少分割掩码抖动,只对视频流有效
                                    min_detection_confidence=0.5,  # 最小检测阈值, 预测结果大于该值保留
                                    min_tracking_confidence=0.5)   # 最小跟踪阈值, 检测结果和跟踪预测结果相似度
        self.grid = np.arange(frame_size[0]*frame_size[1]).reshape(frame_size)  # 网格序号,用于找出mask的区域,从而计算bbox
        self.mask_thr = 0.2   # 分割得分阈值,大于该值为人体像素
        self.thickness = 2
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
         auto_interval=False,                       # 自动跳过相似帧
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
            else:
                raise ValueError(f"Error: path already exist! {save_img_root=}")

            ann_path = Path(save_ann_root).joinpath(Path(video_path).stem)
            if not ann_path.exists():
                ann_path.mkdir(mode=777, parents=True, exist_ok=True)
            else:
                raise ValueError(f"Error: path already exist! {save_ann_root=}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{video_path=}\n{width=}\n{height=}\n{frames_count=}")
        detector = poseDetector(frame_size=(height, width))

        # 初始化前一帧的关键点和边界框
        pos = (30, 30)  # 文字框的左上角坐标
        pre_kpts = [0] * len(pose_dict['request_kpt_indices']) * 3
        pre_bbox = [0, 0, 0, 0]
        for frame_idx in tqdm(range(frames_count), desc=video_path):
            if cap.isOpened():
                _, img = cap.read()
                if frame_idx % interval != 0:
                    continue
                img, results = detector.findPose(img, draw=False)
                kpts, xyxy, area = detector.findPosition(img, results, pose_structure=pose_dict, draw=True)  # landmark
                # 计算姿态相似度，用于去除相似连续帧
                if auto_interval and xyxy != []:
                    iou, pck = pose_similarity(kp1=pre_kpts, kp2=kpts, bbox1=pre_bbox, bbox2=xyxy)
                    if iou > 0.9 and pck > 0.30:
                        continue
                    pre_kpts, pre_bbox = kpts, xyxy
                    cv2.putText(img, str(iou), (pos[0], pos[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.putText(img, str(pck), (pos[0], pos[1]+40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # add info on image
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', '--file', '-f', type=str, help='The path of a video file.')
    parser.add_argument('--root', '--root', '-r', type=str, default='', help='The root directory of video files.')
    parser.add_argument('--video-dir', type=str, default='videos', help='A directory to loading videos')
    parser.add_argument('--interval', '-i', type=int, default=4, help='The interval for frame detection')
    parser.add_argument('--auto-interval', action='store_true', help='Auto skip the images having similar pose with previous images')
    parser.add_argument('--show', action='store_true', help='To visualize the processing for getting pose on each frame.')
    parser.add_argument('--save', '-s', action='store_true', help='To save the visual images and annotations.')
    parser.add_argument('--save-img-root', type=str, default='vis_images', help='A directory to saving visual images')
    parser.add_argument('--save-ann-root', type=str, default='annotations', help='A directory to saving annotations')
    args = parser.parse_args()
    
    """
    目录结构：
    + ./vis_images: 存放视频可视化的帧图像(jpg格式)
    + ./annotations: 存放可视化帧的标注文件(json格式)
    + ./videos: 存放所有待处理的视频
    + ./merge_datasets: 存放合并后的最终数据集
    """

    just_one_video = True if args.root == '' else False  # 是否处理单个视频
    print(f"{args.auto_interval=}")
    if args.auto_interval:
        args.interval = 1

    if just_one_video:
        main(
            video_path=args.file,                # 视频路径
            pose_dict=pose_structure,            # 需要保留和可视化的关键点结构
            interval=args.interval,              # 取帧间隔
            auto_interval=args.auto_interval,    # 自动跳过相似帧
            save=args.save,                      # 是否保存图片和标注
            save_img_root=args.save_img_root,    # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式
            save_ann_root=args.save_ann_root,    # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
            show=args.show                       # 是否显示可视化
        )
    else:
        # 多个视频的根目录，提取到一个目录下
        if args.root != args.video_dir:
            extract_videos(videos_root=args.root, output_dir=args.video_dir)
        video_dir = Path(args.video_dir)
        video_format = ['mov', 'avi', 'mp4','mpg','mpeg','m4v','mkv','wmv']
        for video in video_dir.glob('**/*'):
            if video.suffix.strip('.') in video_format:
                main(
                    video_path=str(video),             # 视频路径
                    pose_dict=pose_structure,          # 需要保留和可视化的关键点结构
                    interval=args.interval,            # 取帧间隔
                    auto_interval=args.auto_interval,  # 自动跳过相似帧
                    save=args.save,                    # 是否保存图片和标注
                    save_img_root=args.save_img_root,  # 保存图片的根目录, 最终目录是 save_img_root/<video_name>, 图片保存为 jpg格式
                    save_ann_root=args.save_ann_root,  # 保存标注文件的更目录, 最终目录是 save_ann_root/<video_name>, 标注保存为json格式
                    show=args.show                     # 是否显示可视化
                )
            print(" Done ".center(50, '-'))

