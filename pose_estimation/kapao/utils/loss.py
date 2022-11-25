# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, num_coords=0,
                 device=None, joint_weights=None):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        # get model device
        device = next(model.parameters()).device if device is None else device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        if self.autobalance:
            self.loss_coeffs = model.module.loss_coeffs if is_parallel(model) else model.loss_coeffs[-1]

        # for k in 'na', 'nc', 'nl', 'anchors':
        #     setattr(self, k, getattr(det, k))
        self.num_coords = num_coords
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors
        self.joint_weights = torch.tensor(joint_weights)  if joint_weights != None else None


    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lkps = torch.zeros(1, device=device)  # keypoint loss
        tcls, tbox, t_kpts, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            img_id, anchor_id, gj, gi = indices[i]  # image, anchor, gridy, gridx
            t_obj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = img_id.shape[0]  # number of targets
            if n:
                # [num_targets, 57], [x, y, w, h, conf, person, c1, ..., c17, x1, x2, ...., x17, y17]
                ps = pi[img_id, anchor_id, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # -0.5~1.5, ignore the grid cell coordinates (cx, cy)
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # range [0, 4] * anchor
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Keypoints
                if self.num_coords:
                    t_kpt = t_kpts[i]
                    vis = t_kpt[..., 2] > 0
                    kpt_vis = t_kpt[vis]
                    if len(kpt_vis):
                        pkp = ps[:, 5 + self.nc:].reshape((-1, self.num_coords // 2, 2))
                        pkp = (pkp.sigmoid() * 4. - 2.) * anchors[i][:, None, :]  # range [-2, 2] * anchor
                        pkp_vis = pkp[vis]
                        l2 = torch.linalg.norm(pkp_vis - kpt_vis[..., :2], dim=-1)  # [batch, num_joints]
                        if self.joint_weights != None:
                            joint_weights = self.joint_weights.tile((vis.shape[0], 1)).to(vis.device)
                            l2 *= joint_weights[vis]
                        lkps += torch.mean(l2)

                # Objectness
                score_iou = iou.detach().clamp(0).type(t_obj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    img_id, anchor_id, gj, gi, score_iou = img_id[sort_id], anchor_id[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                t_obj[img_id, anchor_id, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:5 + self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5 + self.nc], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], t_obj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkps *= self.hyp['kp']

        if self.autobalance:
            loss = (lbox + lobj + lcls) / (torch.exp(2 * self.loss_coeffs[0])) + self.loss_coeffs[0]
            loss += lkps / (torch.exp(2 * self.loss_coeffs[1])) + self.loss_coeffs[1]
        else:
            loss = lbox + lobj + lcls + lkps

        bs = t_obj.shape[0]  # batch size
        return loss * bs, torch.cat((lbox, lobj, lcls, lkps)).detach()

    def build_targets(self, pred, targets):
        """
            pred: list([[Batch, na, h, w, 57], [...], ...]
            targets: tensor([num_gt_person, 57])
            input targets(image_id,class_id,x,y,w,h, x1,y1,v1, ..., x17,y17,v17) == labels from dataset  [57]

        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h, x1,y1,v1, ..., x17,y17,v17)
        num_anchors, num_targets = self.na, targets.shape[0]  # number of anchors, targets
        t_cls, t_box, t_kpts, indices, t_anchors = [], [], [], [], []
        gain = torch.ones(7 + self.num_coords * 3 // 2, device=targets.device)  # normalized to gridspace gain [58]
        anchor_idx = torch.arange(num_anchors, device=targets.device) \
            .float().view(num_anchors, 1).repeat(1, num_targets)  # same as .repeat_interleave(nt)
        # [na * Batch, num_targets, 58] -> 58: [image_id,class_id, x,y,w,h, x1,y1,v1, ..., x17,y17,v17, anchor_id]
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), anchor_idx[:, :, None]), 2)  # append anchor indices

        grid_bias = 0.5  # bias
        offsets = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m == left, top, right, bottom
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * grid_bias  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].to(targets.device)  # [3, 2]
            xy_gain = torch.tensor(pred[i].shape)[[3, 2]]  # w, h
            gain[2:4] = xy_gain  # x, y
            gain[4:6] = xy_gain  # w, h
            for j in range(self.num_coords // 2):
                kpt_idx = 6 + j * 3
                gain[kpt_idx:kpt_idx + 2] = xy_gain

            # Match targets to anchors
            t = targets * gain    # target on grid
            if num_targets:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                keep = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare: joints < 4
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[keep]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                # gxy_inverse = gain[[2, 3]] - gxy  # inverse
                # j, k = ((gxy % 1. < grid_bias) & (gxy > 1.)).T
                # l, m = ((gxy_inverse % 1. < grid_bias) & (gxy_inverse > 1.)).T
                # j = torch.stack((torch.ones_like(j), j, k, l, m))
                left_grid, top_grid = ((gxy % 1. < grid_bias) & (gxy > 1.)).T
                right_grid, bottom_grid = ~left_grid, ~top_grid
                keep = torch.stack((torch.ones_like(left_grid), left_grid, top_grid, right_grid, bottom_grid))
                t = t.repeat((5, 1, 1))[keep]  # only keep 3 dim (0, 1 or 3, 2 or 4) in the first dimension

                _offsets = (torch.zeros_like(gxy)[None] + offsets[:, None])[keep]
            else:
                t = targets[0]
                _offsets = 0

            # Define
            img_idx = t[:, 0].long()  # image
            class_idx = t[:, 1].long()  # class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - _offsets).long()  # int type => grid cell coordinates
            gi, gj = gij.T  # grid xy indices

            if self.num_coords:
                kpt = t[:, 6:-1].reshape(-1, self.num_coords // 2, 3)  # [num_obj, 17, 3]
                kpt[..., :2] -= gij[:, None, :]  # grid kp relative to grid box anchor
                t_kpts.append(kpt)  # [num_obj, 17, 3]

            # Append
            anchor_idx = t[:, -1].long()  # anchor indices
            indices.append((img_idx, anchor_idx, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            t_box.append(torch.cat((gxy - gij, gwh), 1))  # box, [num_obj, 4] [x_offset,y_offset,w,h]
            t_anchors.append(anchors[anchor_idx])  # anchors, [num_obj, 2]
            t_cls.append(class_idx)  # class. [num_obj,]

        return t_cls, t_box, t_kpts, indices, t_anchors

