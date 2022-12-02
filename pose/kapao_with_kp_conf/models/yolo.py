# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os.path
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

try:
    from models.common import *
    from models.experimental import *
    from utils.autoanchor import check_anchor_order
    from utils.general import make_divisible, check_file, set_logging
    from utils.plots import feature_visualization
    from utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
        select_device, copy_attr
except ImportError:
    from pose.kapao_with_kp_conf.models.common import *
    from pose.kapao_with_kp_conf.models.experimental import *
    from pose.kapao_with_kp_conf.utils.autoanchor import check_anchor_order
    from pose.kapao_with_kp_conf.utils.general import make_divisible, check_file, set_logging
    from pose.kapao_with_kp_conf.utils.plots import feature_visualization
    from pose.kapao_with_kp_conf.utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
        select_device, copy_attr

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx = dict(export=False, with_postprocessing=False)  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, num_coords=0):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        # nc = (person, c1, c2, ..., ck, x1,y1,v1, x2,y2,v2, ..., xk,yk,vk),
        # 5 = (xc, yc, w, h, conf),
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.num_coords = num_coords

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx['with_postprocessing'] or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    # print(f"{self.inplace=}")
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    if hasattr(self, 'num_coords') and self.num_coords:
                        y[..., -self.num_coords:] = y[..., -self.num_coords:] * 4. - 2.
                        y[..., -self.num_coords:] *= self.anchor_grid[i].repeat((1, 1, 1, 1, self.num_coords // 2))
                        y[..., -self.num_coords:] += (self.grid[i] * self.stride[i]).repeat((1, 1, 1, 1, self.num_coords // 2))

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
                # z.append(y)

        if self.onnx['export'] and not self.training:
            print(f"{self.onnx=}")
            if self.onnx['with_postprocessing']:
                return torch.cat(z, 1)
            else:
                y = [v.view(-1, self.no) for v in x]  # v[b, na, h, w, 57] -> [b*na*h*w, 57]
                return torch.sigmoid(torch.cat(y, 0))
        else:
            return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None,
                 num_coords=0, autobalance=False):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc + num_coords and nc + num_coords != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc + num_coords}")
            self.yaml['nc'] = nc + (num_coords // 2) * 3  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.sum_params = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        self.num_coords = num_coords
        if autobalance:
            self.loss_coeffs = nn.Parameter(torch.zeros(2))
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.tensor([8., 16., 32.])  # forward for export onnx
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.num_coords = self.num_coords
            m.nc = nc
            self._initialize_biases()  # only run once
            # LOGGER.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False, kp_flip=None,
                scales=[0.5, 1, 2], flips=[None, 3, None]):
        if augment:
            return self.forward_augment(x, kp_flip, s=scales, f=flips)  # augmented inference, None
        return self.forward_once(x, profile, visualize)  # single-scale inference, train

    def forward_augment(self, x, kp_flip, s=[0.5, 1, 2], f=[None, 3, None]):
        img_size = x.shape[-2:]  # height, width
        # s = [1, 0.83, 0.67]  # scales
        # f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        train_out = None
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi, train_out_i = self.forward_once(xi)  # forward
            if si == 1 and fi is None:
                train_out = train_out_i
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size, kp_flip)
            y.append(yi)
        return torch.cat(y, 1), train_out  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                c = isinstance(m, Detect)  # copy input as inplace fix
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_sync()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x

    def _descale_pred(self, p, flips, scale, img_size, kp_flip):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale bbox
            if kp_flip:
                p[..., -self.num_coords:] /= scale  # de-scale kp
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
                if kp_flip:
                    p[..., 6:6 + self.nc - 1] = p[..., 6:6 + self.nc - 1][..., kp_flip]  # de-flip bbox conf
                    p[..., -self.num_coords::2] = img_size[1] - p[..., -self.num_coords::2]  # de-flip kp x
                    p[..., -self.num_coords::2] = p[..., -self.num_coords::2][..., kp_flip]  # swap lr kp (x)
                    p[..., -self.num_coords + 1::2] = p[..., -self.num_coords + 1::2][..., kp_flip]  # swap lr kp (y)

        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5+m.nc] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    sum_params = 0.
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 16)   # 16 for quantization !!!
                # c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        sum_params += np
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    LOGGER.info('\n%12s%10.0f%3s%2.2fM' % ('sum_params: ', sum_params, ' -> ',  sum_params*32/8/1024/1024))

    # self.sum_params = round(sum_params*32/8/1024/1024, 2)
    # if 'plot_sum_params' in globals().keys():  # used to plot model params
    #     global plot_sum_params
    #     plot_sum_params = round(sum_params*32/8/1024/1024, 2)
    sum_params = round(sum_params * 32 / 8 / 1024 / 1024, 2)
    return nn.Sequential(*layers), sorted(save), sum_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--plot', action='store_true',
                        help='plot model params with different depth_multiple and width_multiple')

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    if opt.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        print("plot model params with different depth_multiple and width_multiple")
        with open(opt.cfg, 'r') as fd:
            yaml_dict = yaml.safe_load(fd)
        max_dm = yaml_dict['depth_multiple']
        max_wm = yaml_dict['width_multiple']
        min_dm, min_wm = 0.5, 0.5
        step_size = 0.02

        plot_sum_params = 0  # global vars , change in parse_model
        params = []
        dm = max_dm
        while dm >= min_dm:
            wm = max_wm
            while wm >= min_wm:
                yaml_dict['depth_multiple'], yaml_dict['width_multiple'] = dm , wm
                model = Model(cfg=yaml_dict, nc=18).to(device)
                # print(f"{plot_sum_params=}")
                plot_sum_params = model.sum_params
                wm -= step_size
                del model
                if plot_sum_params in params or \
                    plot_sum_params > 12 or plot_sum_params < 3:   # 3~12M
                    continue
                else:
                    params.append(plot_sum_params)
            dm -= step_size

        print(f"{params=}\n{len(params)=}\n{set(params)=}\n{len(set(params))=}")
        # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        # # figure, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        # x = np.arange(0.4, max_wm, step_size)
        # y = np.arange(0.2, max_dm, step_size)
        # # z = np.array(params)
        # z = np.array(params).reshape((y.size, x.size))
        # plt.imshow(z, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.colorbar()
        # plt.xticks(np.arange(x.size), x)
        # plt.yticks(np.arange(y.size), y)
        # thresh = z.max() / 2.
        # plt.xlim(len(x) - 0.5, 0.5)
        # plt.ylim(len(y) - 0.5, 0.5)
        # for i in range(y.size):
        #     for j in range(x.size):
        #         plt.text(j, i, format(z[i,j], '.2f'),
        #                  horizontalalignment='center',
        #                  color='white' if z[i, j] > thresh else "black")
        # plt.tight_layout()
        # plt.ylabel("depth_multiple")
        # plt.xlabel("width_multiple")
        # plt.savefig("./model_params.pdf", dpi=300)
        # plt.show()
    else:
        model = Model(opt.cfg, nc=18).to(device)
        model.train()
        # model.fuse()
        # model.info(verbose=True)

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
