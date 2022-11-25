import numpy as np
import torch
import cv2
import os
from pathlib import Path
from torchvision import transforms
import random

from object_track.object_track import ObjectTracker
from argparse import ArgumentParser
from classifier.kapao_utils.general import xywh2xyxy, xyxy2xywh


class FallDetector:
    def __init__(self, num_frames, interval_frames, pose_cfg, device, track_type='bot', is_kapao=True):
        self.pose_detector = None
        self.track_detector = None
        self.state_classifier = None
        self.object_sequences = None
        self.is_kapao = is_kapao
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
        self.pose_cfg = pose_cfg
        self.device = device

    def get_pose(self, img, img0):
        if self.is_kapao:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0~255 -> 0.0 ~ 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            output = self.pose_detector(img)
            # person_dets, kp_dets = run_nms(self.pose_cfg, output)
            # bboxes, poses, _, _, _ = post_process_batch(self.pose_cfg, img, [], [[img0.shape[:2]]], person_dets, kp_dets)
        # return bboxes, poses

    def get_track_sequence(self, img, bboxes, xyxy=True):
        bboxes = xywh2xyxy(bboxes) if xyxy else bboxes  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        bboxes, trackIds = self.track_detector.update(bboxes, img)
        # for bbox, track_id in zip(bboxes, trackIds):
        #     x1, y1, x2, y2 = bbox
        return bboxes, trackIds

    def get_state(self, img, seq):
        img = self.transform(img)
        output = self.state_classifier(img, seq)
        state = output.argmax(dim=1)
        return state

    def run(self, path):
        pass

