import sys
from pathlib import Path
from pprint import pprint
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
pprint(sys.path)

import argparse
from pytube import YouTube
import os.path as osp
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages
from models.experimental import attempt_load
import torch
import csv
import cv2
import yaml
import gdown
import imageio
from tqdm import tqdm
from val import run_nms, post_process_batch
from demos.color_config import parse_color_config


def arg_parser():
    parser = argparse.ArgumentParser()
    # video options
    parser.add_argument('-p', '--video-path', default='/home/huangzhiyong/Project/kapao/videos/1.mp4', help='path to video file')
    # parser.add_argument('-p', '--video-path', default='videos/1.mp4', help='path to video file')
    # parser.add_argument('--tag', type=int, default=135, help='stream tag, 137=')
    parser.add_argument('--color', type=int, nargs='+', default=[0,0,255], help='pose color')
    parser.add_argument('--face', action='store_false', help='plot face keypoints')
    parser.add_argument('--face-line', action='store_false', help='plot face line'),
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--kp-obj', action='store_true', help='plot keypoint objects only')
    parser.add_argument('--csv', action='store_true', help='write results so csv file')
    parser.add_argument('--gif', action='store_true', help='create gif')
    parser.add_argument('--fps-size', type=int, default=1)
    parser.add_argument('--gif-size', type=int, nargs='+', default=[480, 270])
    parser.add_argument('--start', type=int, default=34, help='start time (s)')
    parser.add_argument('--end', type=int, default=-1, help='end time (s), -1 for remainder of video')
    parser.add_argument('--kp-size', type=int, default=2, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=3, help='keypoint circle thickness')
    parser.add_argument('--line-thick', type=int, default=4, help='line thickness')
    parser.add_argument('--alpha', type=float, default=0.4, help='pose alpha')

    # model options
    parser.add_argument('--data', type=str, default='/home/huangzhiyong/Project/kapao/data/coco-kp.yaml')
    # parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--weights', default='/home/huangzhiyong/Project/kapao/kapao_weights/kapao_s_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)   # todo ???
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=50)  # todo ???
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    args = parser.parse_args()
    print(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


def get_data_setting(args):
    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False
    return data

def auto_thickness(bbox_area):
    kp_size, kp_thick, line_thick = 0, 0, 2
    bbox_size = bbox_area // 1e3
    if bbox_size < 7:
        kp_size, kp_thick, line_thick = 0, 0, 2
    elif bbox_size < 40:
        kp_size, kp_thick, line_thick = 3, 1, 2
    elif bbox_size < 60:
        kp_size, kp_thick, line_thick = 4, 2, 3
    elif bbox_size < 80:
        kp_size, kp_thick, line_thick = 5, 2, 3
    else:
        kp_size, kp_thick, line_thick = 6, 3, 4
    return kp_size, kp_thick, line_thick

if __name__ == '__main__':
    args = arg_parser()
    data = get_data_setting(args)
    kpt_info, skt_info, name2index = parse_color_config(dataset_name=args.data)
    if 'coco' not in args.data:
        args.face_line = False
        args.face = False

    video_path = args.video_path
    if not video_path:
        raise ValueError("Did not give a video path by '-p your_video_path' !")
    if not osp.isfile(video_path):
        raise  FileNotFoundError(f"{video_path=}")

    device = select_device(args.device, batch_size=1)
    print("Using device: {}".format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    half = args.half & (device.type != 'cpu')
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    img_size = check_img_size(args.imgsz, s=stride)
    dataset = LoadImages(video_path, img_size=img_size, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.end == -1:
        num_frames = int(frame_count - fps * args.start)
    else:
        num_frames = min(int(fps * (args.end - args.start)), frame_count)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    weights_name = args.weights.strip('.').replace('/', '_')
    out_path = '{}_{}_{}'.format(osp.splitext(video_path)[0], osp.splitext(weights_name)[0],
                                 args.device if args.device == 'cpu' else 'gpu')
    print(f"=> {out_path=}")

    if args.csv:
        f = open(out_path + '.csv', 'w')
        csv_writer = csv.writer(f)

    write_video = not args.display and not args.gif
    if write_video:
        writer = cv2.VideoWriter(out_path + '.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    dataset = tqdm(dataset, desc='Running inference', total=num_frames)
    t0 = time_sync()
    for i, (path, img, img0, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()   # uint8 to fp16/32
        img /= 255.0  # 0~255 -> 0.0 ~ 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
        person_dets, kp_dets = run_nms(data, out)
        bboxes, poses, _, _, _ = post_process_batch(data, img, [], [[img0.shape[:2]]], person_dets, kp_dets)

        # im0[433:455, 626:816] = np.mean(im0[434:454, 626:816], axis=(0, 1))  # remove patch
        img0_copy = img0.copy()

        # Draw poses
        csv_row = []
        for j, (bbox, pose, kp_bbox) in enumerate(zip(bboxes, poses, kp_dets)):
            x1, y1, x2, y2 = bbox
            if args.kp_thick < 0 or args.line_thick:
                area = (x2-x1) * (y2 - y1)
                args.kp_size, args.kp_thick, args.line_thick = auto_thickness(area)

            cv2.rectangle(img0_copy, (int(x1), int(y1)), (int(x2), int(y2)), args.color, thickness=args.line_thick)
            if args.csv:
                for x, y, c in pose:
                    csv_row.extend([x, y, c])
            if args.face:
                for joint_idx in data['kp_face']:
                    x, y, c = pose[joint_idx]
                    if not args.kp_obj or c:
                        color = kpt_info[joint_idx]['color']
                        cv2.circle(img0_copy, (int(x), int(y)), args.kp_size, color, args.kp_thick)

            # show keypoints bbox
            right_ankle_bbox = kp_bbox[kp_bbox[:, 5] == 12]
            best_two_bbox_idx = right_ankle_bbox[:, 4].argsort(axis=0)[-2:]
            right_ankle_bbox = right_ankle_bbox[best_two_bbox_idx]
            # print(f"{img.shape=}\t{img0.shape=}\t{right_ankle_bbox.shape=}")
            if right_ankle_bbox.numel() > 0:
                if args.no_kp_dets:
                    right_ankle_bbox[:, :4] = scale_coords(img.shape[2:],   # N,C,H,W
                                                           right_ankle_bbox[:, :4],
                                                           img0.shape[:2])  # H, W, C
                for bbox in right_ankle_bbox:
                    x1, y1, x2, y2, conf = bbox[:5]
                    xc, yc = int((x1+x2)//2), int((y1+y2)//2)
                    cv2.putText(img0_copy, "{:.0f}".format(conf*100), (int(x2), int(y2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                    cv2.circle(img0_copy, (xc, yc), args.kp_size+10, (0,0,255), args.kp_thick)

            left_ankle_bbox = kp_bbox[kp_bbox[:, 5] == 11]
            best_two_bbox_idx = left_ankle_bbox[:, 4].argsort(axis=0)[-2:]
            left_ankle_bbox = left_ankle_bbox[best_two_bbox_idx]
            # print(f"{img.shape=}\t{img0.shape=}\t{right_ankle_bbox.shape=}")
            if left_ankle_bbox.numel() > 0:
                if args.no_kp_dets:
                    left_ankle_bbox[:, :4] = scale_coords(img.shape[2:],   # N,C,H,W
                                                           left_ankle_bbox[:, :4],
                                                           img0.shape[:2])  # H, W, C
                for bbox in left_ankle_bbox:
                    x1, y1, x2, y2, conf = bbox[:5]
                    xc, yc = int((x1+x2)//2), int((y1+y2)//2)
                    cv2.putText(img0_copy, "{:.0f}".format(conf*100), (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
                    cv2.circle(img0_copy, (xc, yc), args.kp_size+7, (0,255,0), args.kp_thick)


            # Skeleton
            num_lines = 19 if args.face_line else 12
            for line_idx in range(num_lines):  # body line 12, if include face it is 19
                kpt_name1, kpt_name2 = skt_info[line_idx]['link']
                kpt_idx1, kpt_idx2 = name2index[kpt_name1], name2index[kpt_name2]
                # if not args.kp_obj or (pose[kpt_idx1, -1] and pose[kpt_idx2, -1]):
                if not args.kp_obj or \
                        (pose[kpt_idx1, -1] > args.conf_thres_kp and pose[kpt_idx2, -1] > args.conf_thres_kp):
                    pt1 = (int(pose[kpt_idx1, 0]), int(pose[kpt_idx1, 1]))
                    pt2 = (int(pose[kpt_idx2, 0]), int(pose[kpt_idx2, 1]))
                    color = skt_info[line_idx]['color']
                    cv2.line(img0_copy, pt1, pt2, color, args.line_thick)

            # Keypoints
            if args.kp_size > 0:
                for x,y,c in pose:
                    # print(f"{c=}")
                    # circle edge without filling
                    cv2.circle(img0_copy, (int(x), int(y)), args.kp_size, (255, 255, 255), args.kp_thick)
                    if c < args.conf_thres_kp: # keypoints with human bbox
                        cv2.circle(img0_copy, (int(x), int(y)), args.kp_size, (0, 0, 255), cv2.FILLED)
                        # continue
                    else:   # keypoints from keypoint bbox
                        cv2.circle(img0_copy, (int(x), int(y)), args.kp_size, (0, 255, 0), cv2.FILLED)

        img0 = cv2.addWeighted(img0, args.alpha, img0_copy, 1-args.alpha, gamma=0)

        t = time_sync() - t0

        if not args.gif and args.fps_size:
            cv2.putText(img0, "{:.1f} FPS".format(1/t), (5*args.fps_size, 25*args.fps_size),
                        cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255),
                        thickness=2 * args.fps_size)
        if args.gif:
            gif_img = cv2.cvtColor(cv2.resize(img0, dsize=tuple(args.gif_size)), cv2.COLOR_RGB2BGR)
            if args.fps_size:
                cv2.putText(gif_img, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                            cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)
            gif_frames.append(gif_img)
        elif write_video:
            writer.write(img0)
        else:
            cv2.imshow('', img0)
            cv2.waitKey(1)

        if args.csv:
            csv_writer.writerow(csv_row)

        t0 = time_sync()
        if i == num_frames - 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    if write_video:
        writer.release()

    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(out_path + '.gif', mode="I", fps=fps) as writer:
            frames_count = len(gif_frames)
            for idx in tqdm(range(frames_count)):
                writer.append_data(gif_frames[idx])
            # for idx, frame in tqdm(enumerate(gif_frames)):
            #     writer.append_data(frame)
    if args.csv:
        f.close()












