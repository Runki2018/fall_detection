import os

import torchvision
import torch
import numpy as np
from pathlib import Path
import cv2
import random

from torchvision import transforms
try:
    from kapao_utils.augmentations import Albumentations, augment_hsv, keypoints_jitter
except ImportError:
    from classifier.kapao_utils.augmentations import Albumentations, augment_hsv, keypoints_jitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def create_cache(data_root, cache_img=False, train_ratio=0.8):
    """TODO: Only work one time"""
    data_root = Path(data_root)
    train_cache = data_root.joinpath("train_cache.npy")
    test_cache = data_root.joinpath("test_cache.npy")
    if train_cache.exists() and test_cache.exists():
        training_set = np.load(str(train_cache), allow_pickle=True)
        test_set = np.load(str(test_cache), allow_pickle=True)
    else:
        # split train/test dataset and create cache
        video_ann_path = []
        for video in data_root.glob("*/Videos/*.mp4"):
            annotation = video.parents[1].joinpath("final_annotations").joinpath(video.stem + '.txt')
            if annotation.exists():
                video_ann_path.append((video, annotation))

        offset = int(len(video_ann_path) * train_ratio)  # split training set and test set, default to 0.8
        random.shuffle(video_ann_path)
        training_set = np.array(video_ann_path[:offset])
        test_set = np.array(video_ann_path[offset:])

        np.save(str(train_cache), training_set)
        np.save(str(test_cache), test_set)
        print(f"save train_cache({len(training_set)},{train_ratio}) => {str(train_cache)}")
        print(f"save test_cache({len(test_set)},{round(1-train_ratio, 3)})  => {str(test_cache)}")

        img_cache_dir = data_root.joinpath("img_cache")
        # if cache_img:
        #     for dataset in [training_set, test_set]:
        #         for video, annotation in zip(dataset):
        #             img_cache_file = img_cache_dir.joinpath()
        #             cap = cv2.VideoCapture(str(video))
        #             success, img = cap.read()
        #             if success:
        #
        #             cap.release()

    return training_set, test_set


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_size=256, num_frame=16, frame_interval=4,
                 is_train=True, train_multi=4, prob_seq_zero=0.2, prob_kp_zero=0.2):
        super().__init__()
        self.num_frame = num_frame
        self.frame_interval = frame_interval
        self.img_size = int(img_size)
        self.training = is_train
        self.train_multi = train_multi
        self.prob_seq_zero = prob_seq_zero
        self.prob_kp_zero = prob_kp_zero
        self.videos, self.annotations = [], []
        for video, ann_file in data_path:
            frames_info = np.loadtxt(str(ann_file), dtype=np.float32)
            if is_train:
                self.videos.append(video)
                self.annotations.append(frames_info)
                self.videos = self.videos
                self.annotations = self.annotations
            else:
                frame_count = len(frames_info)
                max_frame = frame_count - num_frame * frame_interval + 1
                for f1 in range(0, max_frame, 4):
                    f2 = f1 + num_frame * frame_interval
                    self.videos.append(video)
                    self.annotations.append(frames_info[f1:f2:frame_interval])

        self.albumentations = Albumentations() if is_train else None
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.annotations) * self.train_multi if self.training else len(self.annotations)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        # increase the train set, but not increase memory
        item = item % len(self.videos) if self.training else item
        video_path = self.videos[item]
        frame_info = self.annotations[item]

        if self.training:
            # random select 'num_frame' frame to extract time features
            total_frame = len(frame_info)
            max_index = total_frame - self.num_frame * self.frame_interval
            frame_start = random.randint(0, max_index)
            frame_end = frame_start + self.num_frame * self.frame_interval
            frames = frame_info[frame_start:frame_end:self.frame_interval].copy()
        else:
            frames = frame_info.copy()

        # get frame information
        frame_ids = frames[:, 0]
        labels = frames[:, 1]
        width = frames[-1, 2]
        height = frames[-1, 3]
        bbox = frames[:, 4:8]
        keypoints = frames[:, 8:].reshape((self.num_frame, -1, 3))

        # reader = torchvision.io.VideoReader(str(video_path), 'video')
        cap = cv2.VideoCapture(str(video_path))
        last_id = int(frame_ids[-1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_id)
        success, img = cap.read()
        cap.release()
        if not success:
            raise ValueError(f"Read image Error {video_path=}\t{last_id=}")

        img = self.crop_img(img, bbox[-1], height, width)   # Only use the image of last frame
        if self.training:
            img = self.albumentations(img)
            # HSV color-space
            augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)
            bbox *= [width, height, width, height]
            keypoints *= [width, height, 1]
            keypoints = keypoints_jitter(bbox, keypoints, ratio=0.02)
            wh_ratio = bbox[:, 2:4] / [width, height]
        else:
            wh_ratio = bbox[:, 2:4]

        # visualization 1
        # _img_h, _img_w = img.shape[:2]
        # _x1, _y1, _w, _h = bbox[-1]
        # _kpt = keypoints[-1].reshape((-1, 3))
        # for x, y, c in _kpt:
        #     x = int(x * _img_w)
        #     y = int(y * _img_h)
        #     _img = cv2.circle(img, (x, y), 1, (0, 255, 0), cv2.FILLED)
        # cv2.imshow("img", _img)
        # cv2.waitKey()

        keypoints[:, :, 0] = (keypoints[:, :, 0]- bbox[:, 0:1]) / bbox[:, 2:3]  # x - x1 / w
        keypoints[:, :, 1] = (keypoints[:, :, 1] - bbox[:, 1:2]) / bbox[:, 3:4]  # y - y1 / h
        # self.visualization(img, bbox[-1], keypoints[-1], width[-1], height[-1], dw, dh, r)
        keypoints = keypoints[:, :, :2]  # num_seq, num_joints, xyv => num_seq, num_joints, xy
        keypoints = keypoints.reshape((self.num_frame, -1))

        # Limit Dirty Data
        keypoints[keypoints > 1] = 1
        keypoints[keypoints < -1] = -1

        seq = np.concatenate([wh_ratio, keypoints], axis=-1)  # [num_frame, 1 + 3*num_joints]
        if np.any(np.isnan(seq) | np.isinf(seq)) or np.any(seq > 1) or np.any(-seq > 1):
            raise ValueError("Invalid values: {}, {}".format(seq, seq[(seq > 1)|(-seq > 1)]))

        # augmentation
        if self.training:
            if random.random() < self.prob_seq_zero:
                zero_seq = random.randint(0, self.num_frame - 2)  # the last seq is not zero for good
                seq[:zero_seq, :] = 0
            # if random.random() < self.prob_kp_zero:  # TODO: randomly set keypoints to zero

        # did not normalization like kapao
        img = img.astype(np.float32) / 255.
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # img = self.transform(img)
        return img, seq, labels[-1]

    def visualization(self, img, bbox, keypoints, width, height, dw, dh, r):
        x1, y1, w, h = bbox
        w, h = w * width, h * height
        for x, y, v in keypoints:
            x = int(x * w * r + dw)
            y = int(y * h * r + dh)
            img = cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)
        cv2.imshow("img", img)
        cv2.waitKey()

    def crop_img(self, img0, bbox, height, width, color=(114, 114, 114)):
        x1, y1, w, h = bbox * [width, height, width, height]
        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
        img = img0[y1:y2, x1:x2]
        # img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio

        if r != 1:  # if sizes are not equal
            w, h = int(w0 * r), int(h0 * r)
            img = cv2.resize(img, (w, h),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.training else cv2.INTER_LINEAR)
        else:
            w, h = int(w0), int(h0)

        # Compute padding
        dw, dh = (self.img_size - w) / 2, (self.img_size - h) / 2  # wh padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img


def create_dataloader(batch_size, workers, data_path, img_size=256, num_frame=16, frame_interval=2,
                      is_train=True, train_multi=4, prob_seq_zero=0.2, prob_kp_zero=0.2, dist=True):

    dataset = VideoDataset(data_path, img_size, num_frame, frame_interval, is_train, train_multi,
                           prob_seq_zero, prob_kp_zero)
    batch_size = min(batch_size, len(dataset))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers

    loader = torch.utils.data.DataLoader
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = loader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            pin_memory=False,
                            drop_last=False)
    else:
        dataloader = loader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    return dataloader, dataset
