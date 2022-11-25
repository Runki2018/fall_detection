import cv2
import argparse
import numpy as np
import onnxruntime as ort

config = {'person_conf_thres': 0.55, 'person_iou_thres': 0.45,
          'kp_conf_thres': 0.3, 'kp_iou_thres': 0.40, 'conf_thres_kp_person': 0.3,
           # if the center point of kp_bbox is within the range of 'overwrite_tol' pixel, it will replace the predict point
          'overwrite_tol': 12.5,
          'kp_face': [0, 1, 2, 3, 4], 'use_kp_dets': True,
          'segments': {1: [5, 6], 2: [5, 11], 3: [11, 12], 4: [12, 6],
                       5: [5, 7], 6: [7, 9], 7: [6, 8], 8: [8, 10],
                       9: [11, 13], 10: [13, 15], 11: [12, 14], 12: [14, 16], },
          # 'crowd_segment': {1:[0, 13], 2:[1, 13], 3:[0, 2], 4:[12, 6], 5: [5, 7],
          #                   6:[7, 9], 7:[6, 8], 8:[8, 10], },
          # 'crowd_kp_face': [],
          'input_size': (640, 640),
          'anchors': [[10, 13, 16, 30, 33, 23],  # P3/8
                      [30, 61, 62, 45, 59, 119],  # P4/16
                      [116, 90, 156, 198, 373, 326]],  # P5/32
          'stride': [8., 16., 32.]  # grid size
          }


class kapao():  # Keypoint and Pose as Object
    def __init__(self, model_path):
        with open('./cpp_detect/class.names', 'rt') as fd:
            self.classes = fd.read().rstrip('\n').split('\n')
        self.lines = config['segments']
        self.kp_face = config['kp_face']

        self.num_classes = len(self.classes)
        self.input_height, self.input_width = config['input_size']
        anchors = config['anchors']
        self.stride = np.array(config['stride'])
        self.nl = len(anchors)  # number of heads
        self.na = len(anchors[0]) // 2  # number of anchors per head
        self.grid = [np.zeros(1)] * self.nl
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        # self.net = cv2.dnn.readNetFromONNX(model_path)  # TODO: whatt?
        self.net = cv2.dnn.readNet(model_path)  # TODO: whatt?
        self._input_names = 'images'
        self.last_ind = 5 + self.num_classes  # total index: 5(x,y,w,h,conf) + num_classes + 2*num_kpts

    def resize_image(self, src_img, keep_ratio=True, dynamic=False):
        top, left, h_new, w_new = 0, 0, self.input_height, self.input_width
        if keep_ratio and src_img.shape[0] != src_img.shape[1]:
            hw_scale = src_img.shape[0] / src_img.shape[1]  # h / w
            if hw_scale > 1:
                h_new, w_new = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(src_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    left = int((self.input_width - w_new) * 0.5)
                    img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - w_new,
                                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
            else:
                h_new, w_new = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(src_img, (w_new, h_new), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    top = int((self.input_height - h_new) * 0.5)
                    img = cv2.copyMakeBorder(img, top, self.input_height - h_new - top, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            img = cv2.resize(src_img, (self.input_width, self.input_height), inperpolation=cv2.INTER_AREA)
        return img, h_new, w_new, top, left

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, pad_size=None):
        """"""
        frame_height, frame_width = frame.shape[:2]
        h_new, w_new, h_pad, w_pad = pad_size
        h_ratio, w_ratio = frame_height / h_new, frame_width / w_new

        # 1. Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        person_confidences, kp_confidences = [], []  # confidences
        person_boxes, kp_boxes = [], []  # bboxes
        person_classsIds, kp_classIds = [], []  # class indices
        person_rowins = []
        for i in range(outs.shape[0]):
            detection = outs[i, :]
            x, y, w, h, obj_conf = detection[:5]
            scores = detection[5: self.last_ind]
            classId = np.argmax(scores)
            confidence = scores[classId] * obj_conf  # class_scores * object_confidence
            if classId == 0:
                if obj_conf > config['person_conf_thres'] and confidence > config['person_conf_thres']:
                    center_x = int((x - w_pad) * w_ratio)
                    center_y = int((y - h_pad) * h_ratio)
                    width = int(w * w_ratio)
                    height = int(h * h_ratio)
                    left = max(int(center_x - width * 0.5), 0)
                    top = max(int(center_y - height * 0.5), 0)

                    person_confidences.append(float(confidence))
                    person_boxes.append([left, top, width, height])
                    person_classsIds.append(classId)
                    person_rowins.append(i)
            else:
                if obj_conf > config['kp_conf_thres'] and confidence > config['kp_conf_thres']:

                    center_x = int((x - w_pad) * w_ratio)
                    center_y = int((y - h_pad) * h_ratio)
                    width = int(w * w_ratio)
                    height = int(h * h_ratio)
                    left = max(int(center_x - width * 0.5), 0)
                    top = max(int(center_y - height * 0.5), 0)

                    kp_confidences.append(confidence)
                    kp_boxes.append([left, top, width, height])
                    kp_classIds.append(classId)

        # 2. Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        person_indices = cv2.dnn.NMSBoxes(person_boxes, person_confidences, config['person_conf_thres'],
                                          config['person_iou_thres']).flatten()
        kp_indices = cv2.dnn.NMSBoxes(kp_boxes, kp_confidences, config['kp_conf_thres'],
                                      config['kp_iou_thres']).flatten()
        poses = []  # keypoints
        for i in person_indices:
            if person_confidences[i] > config['conf_thres_kp_person']:
                pose = outs[person_rowins[i], self.last_ind:].reshape((-1, 2))  # [num_joints, 2]
                pose[:, 0] = (pose[:, 0] - w_pad) * w_ratio
                pose[:, 1] = (pose[:, 1] - h_pad) * h_ratio
                poses.append(pose)
        nd = len(poses)
        poses = np.array(poses)  # [nd, num_joints, 2]
        poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)  # [nd, num_joints, 3]
        # fuse keypoints bbox and keypoints
        for j in kp_indices:
            box = kp_boxes[j]
            x = box[0] + 0.5 * box[2]
            y = box[1] + 0.5 * box[3]
            pt_id = kp_classIds[j] - 1  # 1~num_joints -> 0 ~ num_joints - 1
            pose_kps = poses[:, pt_id, :]
            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
            kp_match = np.argmin(dist)
            # TODO: why we need to use pose_kps[kp_match, 2], which is always zero 0.
            if kp_confidences[j] > pose_kps[kp_match, 2] and dist[kp_match] < config['overwrite_tol']:
                poses[kp_match, pt_id, :] = np.array([x, y, kp_confidences[j]])

        # 3. Visualization: draw line and bbox
        for i in person_indices:
            left, top, width, height = person_boxes[i]
            frame = self.drawPred(frame, person_classsIds[i], person_confidences[i],
                                  left, top, left + width, top + height)
        for pose in poses:
            for seg in self.lines.values():
                pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 1)
            for x, y, c in pose:
                if c > 0:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 1)
            # TODO: add args to control whether to draw face keypoints
            # for x, y, c, in pose[self.kp_face]:
            #     cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 255), 1)
        # TODO: add args to control weather to draw keypoint bounding boxes
        # for i in kp_indices:
        #     left, top, width, height = kp_boxes[i]
        #     frame = self.drawPred(frame, kp_classIds[i], kp_confidences[i],
        #                           left, top, left+width, top+height)
        return frame

    def drawPred(self, frame, classId, conf, x1, y1, x2, y2):
        # 1. Draw a bounding box. (x_left, y_top, x_right, y_bottom) = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # 2. Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1 = max(y1, labelSize[1])  # top
        # cv2.rectangle(frame, (x1, y1-round(1.5*labelSize[1])), (x1+round(1.5*labelSize[0]), y1+baseLine),
        #               (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

    def detect(self, src_img):
        img, h_new, w_new, h_pad, w_pad = self.resize_image(src_img)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        # 1. Set the input to the network
        self.net.setInput(blob, self._input_names)

        # 2. Runs the forward pass to get output of the output layers
        # TODO: check the dimensions of outs [na*(w1*h1+w2*h2+w3*h3), 6 + 3*num_kp]
        # outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)
        outs = self.net.forward()  # [25200, 57]
        # outs = 1 / (1 + np.exp(-outs))  # sigmoid

        # 3. Decode inference output
        ig = 0  # the index of beginning grid
        # outs = np.zeros_like(outs)
        # outs[0, :6] = [0, 0, 0.5, 0.5, 1, 1]
        # outs[23000, :6] = [0, 0, 0.5, 0.5, 1, 1]
        # outs[2300, :7] = [0, 0, 0.5, 0.5, 1, 0, 1]
        for i in range(len(outs)):
            # if outs[i, 4] > config['person_conf_thres']:
            with open("./detect_results.txt", 'a+') as fd:
                line = list(outs[i, :])
                line = ["%.3f" % x for x in line]
                line = " ".join(line) + ' '
                fd.write(line)

        for i in range(self.nl):
            h, w = int(self.input_height / self.stride[i]), int(self.input_width / self.stride[i])
            length = int(self.na * h * w)  # the number of grids in an output layer
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)
            # In COCO dim-1: [tx, ty, tw, th, conf, c_person, c1, c2, ...,c17, x1, y1, ..., x17, y17]
            xy = outs[ig:ig + length, 0:2]  # the coordinates of bounding box center
            wh = outs[ig:ig + length, 2:4]  # the width and height of bounding box
            kp = outs[ig:ig + length, self.last_ind:]  # the coordinates (x1, y1, .., xn, yn) of keypoints
            # # TODO: check this, because of not using sigmoid to decode outputs
            # yolo.py has done the post-process to decode the output
            xy = int(self.stride[i]) * (2. * xy - 0.5 + np.tile(self.grid[i], (self.na, 1)))
            wh = (2. * wh) ** 2 * np.repeat(self.anchor_grid[i], h * w, axis=0)
            num_kp = (outs.shape[1] - self.last_ind) // 2  # len([x1, y1, ..., xn, yn])  // 2
            kp = 4. * kp - 2.
            kp *= np.tile(np.repeat(self.anchor_grid[i], h * w, axis=0), (1, num_kp))
            kp += np.tile(np.tile(self.grid[i], (self.na, 1)) * int(self.stride[i]), (1, num_kp))

            outs[ig:ig + length, 0:2] = xy
            outs[ig:ig + length, 2:4] = wh
            outs[ig:ig + length, self.last_ind:] = kp

            ig += length
        src_img = self.postprocess(src_img, outs, pad_size=(h_new, w_new, h_pad, w_pad))
        return src_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--img-path', type=str, default='data/data/coco/images/test2017/000000000057.jpg',
    parser.add_argument('--img-path', type=str,
                        # default='data/data/coco/images/test2017/000000000057.jpg',
                        default="/home/huangzhiyong/Project/kapao/res/crowdpose_100024.jpg",
                        help='image path')
    parser.add_argument('--model-path', type=str, default='./cpp_detect/model.onnx')
    args = parser.parse_args()

    print(f"{args.model_path}")
    # session = ort.InferenceSession(args.model_path)
    # img0 = cv2.imread(args.img_path)
    # img1 = cv2.resize(img0, (640, 640))
    # ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # ori_image = ori_image.astype(np.float32) / 255.0
    # img_input = np.transpose(ori_image, [2, 0, 1])[None]
    # results = session.run(['output'], {'images': img_input})[0]
    # print(f"{type(results)=}")
    # print(f"{results.shape=}")
    model = kapao(args.model_path)
    img_input = cv2.imread(args.img_path)
    print(f"{img_input.shape=}")
    img_input = model.detect(img_input)
    print(f"{img_input.shape=}")
    # img_input = model.drawPred(img_input, 0, 1, 300, 300, 350, 350)

    window_name = "Keypoint and Pose as Object (KAPAO)"
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, img_input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
