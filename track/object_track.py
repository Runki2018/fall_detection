import time
from argparse import ArgumentParser
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
from random import randint

try:
    from sort import Sort
    from bot_sort.bot_sort import BoTSORT
    from oc_sort.ocsort import OCSort
except ImportError:
    from track.sort import Sort
    from track.bot_sort.bot_sort import BoTSORT
    from track.oc_sort.ocsort import OCSort

np.random.seed(0)

def parse_args():
    parser = ArgumentParser(description="Object Track")
    parser.add_argument('-t', type=str, default='bot',
                        help='the type of object tracker, including <sort>, <bot>, <oc>.')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--out_path", help="Path to results.", type=str, default='output')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')

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
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true", help="fuse score and iou for association")
    # bot => CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--ablation", action='store_true', help="GMC verbose ablation")
    parser.add_argument("--name", type=str, default=None, help="GMC verbose model name")

    # oc
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    opts = parser.parse_args()
    return opts

def _int(a, b):
    return int(a), int(b)

class Display:
    def __init__(self, display, out_path, mot_path='mot_benchmark'):
        self.display = display
        self.mot_path = Path(mot_path)
        if display:
            if not self.mot_path.exists():
                print(
                    '\n\tERROR: mot_benchmark link not found!\n\n'
                    'Create a symbolic link to the MOT benchmark\n'
                    '(https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n'
                    '$ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
                exit()
            # cv2.namedWindow("display", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("display", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("display", 640, 640)
            self.img = None
            self.out_path = out_path
            # self.colors = np.random.randint(0, 255, (32, 3)) # used only for display
            self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(32)]

    def imread(self, phase, seq_name, frame):
        img_file = self.mot_path / phase / seq_name / 'img1' / f'{frame:06d}.jpg'
        self.img = cv2.imread(str(img_file))
        return self.img

    def show(self):
        if not self.display:
            return

        cv2.imshow("display", self.img)
        key = cv2.waitKey(1)
        if key & 0xFFFF == ord('q'):
            exit()

    def draw_bbox(self, x1, y1, x2, y2, t_id):
        if not self.display:
            return
        t_id = int(t_id)
        cv2.rectangle(self.img, _int(x1, y1), _int(x2, y2),
                      color=self.colors[t_id % 32], thickness=2)
        cv2.putText(self.img, str(t_id), _int(x1, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


class ObjectTracker:
    def __init__(self, args, frame_rate=30):
        assert args.t in ['sort', 'oc', 'bot']  # sort, ocsort, botsort
        self.args = args
        if self.args.t == 'sort':
            self.obj_tracker = Sort(max_age=args.max_age,
                                   min_hits=args.min_hits,
                                   iou_threshold=args.iou_threshold)
        elif self.args.t == 'bot':
            self.obj_tracker = BoTSORT(args, frame_rate=frame_rate)
        elif self.args.t == 'oc':
            self.obj_tracker = OCSort(det_thresh=args.track_high_thresh,
                                      iou_threshold=args.iou_threshold,
                                      use_byte=args.use_byte)
        else:
            raise ValueError("ObjectTracker only support SORT, OC-SORT, and BoT-SORT.")

    def update(self, detections, img=None):
        """
        :param detections: num_bbox * [x1, y1, x2, y2, score]
        :param img: cv2.imread
        :return: bbox list[num_bbox, 4], track_id list[num_bbox]
        """
        bbox, track_id = [], []
        if self.args.t == 'sort':
            online_targets = self.obj_tracker.update(detections)
            online_targets = np.array(online_targets)
            bbox = online_targets[:, :4].tolist()
            track_id = online_targets[:, 4].astype(np.int32).tolist()

        elif self.args.t == 'bot':
            online_targets = self.obj_tracker.update(detections, img)
            for target in online_targets:
                tlwh = target.tlwh
                tid = target.track_id
                # score = target.score
                x1, y1, w, h = tlwh
                vertical = w / h > self.args.aspect_ratio_thresh
                if w * h > self.args.min_box_area and not vertical:
                    bbox.append([x1, y1, x1+w, y1+h])  # x1, y1, x2, y2
                    track_id.append(int(tid))

        elif self.args.t == 'oc':
            # height, width = img.shape[:2]
            online_targets = self.obj_tracker.update(detections)
            for target in online_targets:
                x1, y1, x2, y2, tid = target[:5]
                w, h = x2 - x1, y2 - y1
                vertical = w / h > self.args.aspect_ratio_thresh
                if w * h > self.args.min_box_area and not vertical:
                    bbox.append([x1, y1, x2, y2])  # x1, y1, x2, y2
                    track_id.append(int(tid))

        return bbox, track_id


def main():
    args = parse_args()
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    display = Display(args.display, args.out_path)
    assert args.t in ['sort', 'oc', 'bot']  # sort, ocsort, botsort

    out_path = Path(args.out_path)
    if not out_path.exists():
        out_path.mkdir(mode=0o777, parents=True, exist_ok=True)

    # path to the detect result
    pattern = Path(args.seq_path) / phase
    for seq_dets_fn in pattern.glob("*/det/det.txt"):
        obj_tracker = ObjectTracker(args, frame_rate=30)  # TODO: confirm frame_rate
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq_name = seq_dets_fn.parts[2]
        out_file = out_path.joinpath(seq_name + '.txt')
        with out_file.open('w') as fd:
            print("Processing %s." % (seq_name))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1 # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                img = display.imread(phase, seq_name, frame)
                start_time = time.time()
                bbox_list, track_id_list = obj_tracker.update(dets, img)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for bbox, t_id in zip(bbox_list, track_id_list):
                    x1, y1, x2, y2 = bbox
                    print(f"{frame}, {int(t_id)},"
                          f" {x1:.2f}, {x2:.2f}, {x2-x1:.2f}, {y2-y1:.2f}",
                          file=fd)

                    display.draw_bbox(x1, y1, x2, y2, t_id)
                display.show()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if args.display:
        print("Note: to get real runtime results run without the option: --display")


if __name__ == '__main__':
    main()
