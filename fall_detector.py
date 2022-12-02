import numpy as np
import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import random
from collections import defaultdict

from track.object_track import ObjectTracker
from argparse import ArgumentParser
from classifier.kapao_utils.general import xywh2xyxy, xyxy2xywh
from classifier.models.classifer import Classifer

from pose.kapao_with_kp_conf.val import run_nms, post_process_batch
from pose.kapao_with_kp_conf.utils.datasets import LoadImages, IMG_FORMATS, VID_FORMATS
from pose.kapao_with_kp_conf.utils.general import check_img_size, scale_coords
from pose.kapao_with_kp_conf.utils.metrics import box_iou
from pose.kapao_with_kp_conf.demos.color_config import parse_color_config


def _int(*args):
    if len(args) == 1 and isinstance(args, (list, tuple)):
        return [int(a) for a in args[0]]
    else:
        return [int(a) for a in args]

def arg_parser():
    parser = ArgumentParser()
    # video options
    # parser.add_argument('-p', '--input-path', default='classifier/data/le2i_falldataset/Coffee_room_01/Videos/video (16).mp4', help='path to video file')
    parser.add_argument('-p', '--input-path', default='track/mot_benchmark/test/ETH-Jelmoli/img1', help='path to video file')
    # parser.add_argument('-p', '--input-path', default='track/mot_benchmark/train/TUD-Campus/img1', help='path to video file')
    parser.add_argument('--out-path', type=str, default='out_videos/', help='path to results video')
    parser.add_argument('--color', type=int, nargs='+', default=[0,0,255], help='pose color')
    parser.add_argument('--pose-pth', type=str,  default='jit_models/kapao2.pt', help='pose color')
    parser.add_argument('--cls-pth', type=str, default='jit_models/cls.pt', help='pose color')

    # parser.add_argument('--face', action='store_false', help='plot face keypoints')
    # parser.add_argument('--face-line', action='store_false', help='plot face line'),
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--kp-obj', action='store_true', help='plot keypoint objects only')
    parser.add_argument('--fps-size', type=int, default=1)
    parser.add_argument('--gif-size', type=int, nargs='+', default=[480, 270])
    parser.add_argument('--start', type=int, default=0, help='start time (s)')
    parser.add_argument('--end', type=int, default=-1, help='end time (s), -1 for remainder of video')
    parser.add_argument('--kp-size', type=int, default=2, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=3, help='keypoint circle thickness')
    parser.add_argument('--line-thick', type=int, default=4, help='line thickness')

    # model options
    parser.add_argument('--data', type=str, default='/home/huangzhiyong/Project/kapao/data/coco-kp.yaml')
    # parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=512)
    # parser.add_argument('--weights', default='/home/huangzhiyong/Project/kapao/kapao_weights/kapao_s_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)   # todo ???
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=50)  # fuse keypoints and keypoint bbox within this distance
    parser.add_argument('--swap-nms', action='store_false', help='NMS each pair of keypoints, such as left-ankle and right-ankle.')
    parser.add_argument('--vis-conf-values', action='store_true', help='display the confidence values')
    parser.add_argument('--vis-kp-conf', type=float, default=0.5, help='')
    parser.add_argument('--ow-ratios', type=float, nargs='+', default=[0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    # object track
    parser.add_argument('-t', type=str, default='bot',
                        help='the type of object tracker, including <sort>, <bot>, <oc>.')
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
    #                     action='store_true')
    # SORT => tracking args
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)

    # bot => tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=2.5,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true",
                        help="fuse score and iou for association")
    # bot => CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--ablation", action='store_true', help="GMC verbose ablation")
    parser.add_argument("--name", type=str, default=None, help="GMC verbose model name")

    # oc
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")

    # classifier
    parser.add_argument("--num_frames", type=int, default=16,  help="num_frames")
    parser.add_argument("--interval_frames", type=int, default=4, help="interval_frames")

    args = parser.parse_args()
    print(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


class FallDetector:
    def __init__(self, device='cpu'):
        self.pose_detector = None
        self.track_detector = None
        self.state_classifier = None
        self.object_sequences = None
        self.device = torch.device(device)
        self.crop_size = 128
        self.to_tensor = transforms.ToTensor()
        print("Using device: {}".format(device))
        args = arg_parser()
        self.args = args
        self.pose_cfg = dict(
            nc=13,
            num_coords=24,      # 2 * num_keypoints
            nl=3,  # num_head of kapao
            stride=[8, 16, 32],  # feature stride for each head
            imgsz=args.imgsz,   # input image size of pose detector
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            use_kp_dets= not args.no_kp_dets,
            conf_thres_kp=args.conf_thres_kp,
            iou_thres_kp=args.iou_thres_kp,
            conf_thres_kp_person=args.conf_thres_kp_person,
            overwrite_tol=args.overwrite_tol,
            scales=args.scales,
            flips=[None if f == -1 else f for f in args.flips],
            count_fused=False,
            swap_nms=args.swap_nms,
            ow_ratios=args.ow_ratios
        )
        self.num_frames = args.num_frames
        self.interval_frames = args.interval_frames

    def get_pose(self, img, img0):
        output = self.pose_detector(img)

        person_dets, kp_dets = run_nms(self.pose_cfg, output[None])
        bboxes, poses, scores, _, _ = post_process_batch(self.pose_cfg,
                                                         img, [], [[img0.shape[:2]]],
                                                         person_dets, kp_dets)
        if len(bboxes) > 0:
            bboxes = np.stack(bboxes, axis=0)  # [N, 4], (x1, y1, x2, y2)
            poses = np.stack(poses, axis=0)    # [N, 24]
            scores = np.array(scores).reshape((-1, 1))   # [N, 1]
            bboxes = np.concatenate([bboxes, scores], axis=1)  # [N, 5]
        return bboxes, poses

    def get_track_sequence(self, img, bboxes, xyxy=True):
        bboxes = np.array(bboxes)
        bboxes = xywh2xyxy(bboxes) if not xyxy else bboxes  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        bboxes, trackIds = self.track_detector.update(bboxes, img)
        return bboxes, trackIds

    def get_state(self, img, seq):
        img = self.transform(img)
        output = self.state_classifier(img, seq)
        state = output.argmax(dim=1)
        return state

    def load_model(self):
        args = self.args
        self.args.half = args.half & (self.device.type != 'cpu')

        # self.pose_detector = attempt_load(args.pose_weights, map_location=device)  # load FP32 model
        self.pose_detector = torch.jit.load(args.pose_pth, map_location=self.device)
        if self.args.half:  # half precision only supported on CUDA
            self.pose_detector.half()

        # TODO: add Conv-BN fuse
        self.state_classifier = torch.jit.load(args.cls_pth, map_location=self.device) # load FP32 model
        if self.args.half:  # half precision only supported on CUDA
            self.state_classifier.half()

        self.pose_detector.eval()
        self.state_classifier.eval()

    def get_out_file(self):
        self.args.input_path  = Path(self.args.input_path)
        out_path = Path(self.args.out_path)
        if not out_path.exists():
            out_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        if self.args.input_path.is_dir():
            return out_path.joinpath(self.args.input_path.parts[-1] + '.mp4')
        else:
            return out_path.joinpath(self.args.input_path.name)

    def transform(self, img, to_tensor=False):
        if to_tensor:
            img = img.astype(np.float32) / 255.
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.to(self.device)
        img = img.half() if self.args.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0~255 -> 0.0 ~ 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return  img

    def match(self, bboxes, poses, bboxes_t):
        box1 = torch.tensor(bboxes)[:, :4]
        box2 = torch.tensor(bboxes_t)[:, :4]
        iou = box_iou(box1, box2)
        index = iou.argmax(dim=1).numpy()
        bboxes = bboxes[index]
        poses = poses[index]
        return bboxes, poses

    def crop_img(self, img0, bbox, color=(114, 114, 114)):
        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = _int(x1, y1, x2, y2)
        img = img0[y1:y2, x1:x2]

        h0, w0 = img.shape[:2]  # orig hw
        r = self.crop_size / max(h0, w0)  # ratio

        if r != 1:  # if sizes are not equal
            w, h = int(w0 * r), int(h0 * r)
            img = cv2.resize(img, (w, h),
                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        else:
            w, h = int(w0), int(h0)

        # Compute padding
        dw, dh = (self.crop_size - w) / 2, (self.crop_size - h) / 2  # wh padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img

    def vis_pose(self, img, bbox, pose, tid):
        args = self.args
        x1, y1, x2, y2 = bbox[:4]
        kpt_info, skt_info, name2index = parse_color_config(dataset_name="mixed")
        cv2.rectangle(img, _int(x1, y1), _int(x2, y2), args.color, thickness=args.line_thick)
        cv2.putText(img, str(tid), _int(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Skeleton
        num_lines = len(skt_info)
        for line_idx in range(num_lines):  # body line 12, if include face it is 19
            kpt_name1, kpt_name2 = skt_info[line_idx]['link']
            kpt_idx1, kpt_idx2 = name2index[kpt_name1], name2index[kpt_name2]
            # if not args.kp_obj or (pose[kpt_idx1, -1] and pose[kpt_idx2, -1]):
            if not args.kp_obj and \
                    (pose[kpt_idx1, -1] > args.vis_kp_conf and pose[kpt_idx2, -1] > args.vis_kp_conf):
                pt1 = (int(pose[kpt_idx1, 0]), int(pose[kpt_idx1, 1]))
                pt2 = (int(pose[kpt_idx2, 0]), int(pose[kpt_idx2, 1]))
                color = skt_info[line_idx]['color']
                cv2.line(img, pt1, pt2, color, args.line_thick)

        # Keypoints
        if args.kp_size > 0:
            for i, (x, y, c) in enumerate(pose, start=1):
                if args.vis_conf_values and args.kp_thick > 1 and i in [5, 6, 11, 12]:
                    cv2.putText(img, "{:.0f}".format(c * 100), (int(x + 10), int(y + 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=1)

                cv2.circle(img, (int(x), int(y)), args.kp_size, (255, 255, 255), args.kp_thick)
                if c < args.vis_kp_conf:  # keypoints with human bbox
                    # cv2.circle(img0_copy, (int(x), int(y)), args.kp_size, (0, 0, 255), cv2.FILLED)
                    continue
                else:  # keypoints from keypoint bbox
                    cv2.circle(img, (int(x), int(y)), args.kp_size, (0, 255, 0), cv2.FILLED)
        return img

    def vis_state(self, img, bbox, state_id):
        x1, y1, x2, y2 = bbox[:4]
        cv2.putText(img, str(state_id), _int(x2-10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        return img

    def run(self):
        """
        file_path: image/video file or image/video directory
        """
        args = self.args
        self.load_model()

        # load dataset
        # stride = int(self.pose_detector.stride.max())  # model stride
        img_size = check_img_size(args.imgsz, s=32)  # stride = 32
        dataset = LoadImages(args.input_path, img_size=args.imgsz, stride=32, auto=True)
        if self.device.type != 'cpu':
            self.pose_detector(torch.zeros(1, 3, img_size, img_size)\
                               .to(self.device).type_as(next(self.pose_detector.parameters())))  # run once

        # get dataset statues
        if dataset.cap is not None:
            cap = dataset.cap
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.end == -1:
                num_frames = int(frame_count - fps * args.start)
            else:
                num_frames = min(int(fps * (args.end - args.start)), frame_count)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            w, h = None, None
            num_frames = dataset.__len__()
            fps = 30

        self.track_detector = ObjectTracker(self.args, frame_rate=fps)
        out_file = self.get_out_file()
        print(f"write video to {str(out_file)}")

        if not (w is None or h is None):
            writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        else:
            writer = None
        dataset = tqdm(dataset, desc='Running inference', total=num_frames)
        seq_dict = defaultdict(list)
        img_dict = defaultdict(list)
        bbox_dict = dict()

        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        for i, (path, img, img0, _) in enumerate(dataset):
            if writer is None:
                writer = cv2.VideoWriter(str(out_file),
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (img0.shape[1], img0.shape[0]))
            img = self.transform(img)
            bboxes, poses = self.get_pose(img, img0)
            if len(bboxes) > 0:
                bboxes_t, track_ids = self.get_track_sequence(img0, bboxes, xyxy=True)
                if len(track_ids) > 0:
                    bboxes, poses = self.match(bboxes, poses, bboxes_t)
                    img0_h, img0_w = img0.shape[:2]
                    for bbox, pose, tid in zip(bboxes, poses, track_ids):
                        bbox_dict[tid] = bbox
                        img0 = self.vis_pose(img0, bbox, pose, tid)
                        x1, y1, x2, y2, score = bbox
                        img_crop = self.crop_img(img0, bbox)
                        img_crop = self.transform(img_crop, to_tensor=True)
                        w, h = x2 - x1, y2 - y1
                        pose[:, 0] = (pose[:, 0] - x1) / w
                        pose[:, 1] = (pose[:, 1] - y1) / h

                        wh_ratio = np.array([w, h]) / [img0_w, img0_h]
                        pose = pose[:, :2]
                        seq = np.concatenate([wh_ratio, pose.flatten()], axis=0)
                        seq[seq > 1] = 1
                        seq[seq < -1] = -1
                        # if np.any(np.isnan(seq) | np.isinf(seq)) or np.any(seq > 1) or np.any(-seq > 1):
                        #     raise ValueError("Invalid values: {}, {}".format(seq, seq[(seq > 1)|(-seq > 1)]))
                        seq = torch.tensor(seq, device=self.device).unsqueeze(dim=0)

                        seq_dict[tid].append(seq)
                        img_dict[tid].append(img_crop)

                    for tid in track_ids:
                        seq = seq_dict[tid][-self.num_frames*self.interval_frames::self.interval_frames]
                        img_crop = img_dict[tid][-1]
                        seq_16 = torch.zeros((1, self.num_frames, 26), device=self.device)
                        seq_16[0,-len(seq):] = torch.cat(seq, dim=0)
                        state = self.state_classifier(img_crop, seq_16)
                        # print(f"{state=}\t{torch.sigmoid(state)=}")
                        state = state.argmax().item()
                        self.vis_state(img0, bbox_dict[tid], state)

            cv2.imshow("image", img0)
            key = cv2.waitKey(1)
            if key & 0xFFFF == ord('q'):
                break

            writer.write(img0)
        cv2.destroyAllWindows()
        writer.release()

if __name__ == '__main__':
    fd = FallDetector()
    fd.run()
