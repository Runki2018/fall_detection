import os
import re
import glob
import math
import torch
import urllib
import random
import numpy as np
from pathlib import Path
from kapao_utils.torch_utils import init_torch_seeds
import matplotlib.pyplot as plt
import sys


def fitness(results, b=1):
    """

    :param results: (tuple) => (precision, recall)
    :b: (float) a factor for trade-off between precision and recall
    :return: mean F_beta score (F1 score with beta factor)
    """
    F_beta = []  # F1 for each class
    precision, recall = results
    for p, r in zip(precision, recall):
        numerator = (1+b**2) * p * r
        denominator =(b**2) * p + r
        F_beta.append(numerator / (denominator + 1e-6))
    mF_beta = sum(F_beta) / len(F_beta)
    return mF_beta

def plot_confusion_matrix(confusion_matrix, out_path, class_names):
    """
        compute metric from confusion matrix and
         save corresponding file to specified directory (out_path)
    :param confusion_matrix:  Tensor([num_class, num_class])
    :param out_path: (str) output path for saving confusion matrix information
    :param class_names: (list) class names ['class name1', 'class name2', ...]
    :return:
    """
    confusion_matrix = confusion_matrix.detach().cpu().numpy()
    diagonal = confusion_matrix.diagonal(offset=0)
    rows_sum = confusion_matrix.sum(axis=1)
    cols_sum = confusion_matrix.sum(axis=0)
    precision = [round(acc*100, 3) for acc in diagonal/(rows_sum + 1e-6)]
    recall = [round(r*100, 3) for r in diagonal/(cols_sum + 1e-6)]

    out_path = out_path if isinstance(out_path, Path) else Path(out_path)
    record_file = out_path / "confusion_matrix.log"
    with record_file.open('a+') as fd:
        for out_stream in [fd, sys.stdout]:
            print("-"*100, file=out_stream)
            print("predicted samples of each class: \t{}".format(rows_sum), file=out_stream)
            print("actual samples of each class:\t\t{}".format(cols_sum), file=out_stream)
            print("correct count of each class:\t\t{}".format(diagonal), file=out_stream)
            print(f"precision of each class:\t\t{precision}", file=out_stream)
            print(f"recall of each class:\t\t\t{recall}", file=out_stream)

    # plot confusion matrix
    plot_file = str(out_path / 'confusion_matrix.png')
    num_class = confusion_matrix.shape[0]
    thresh = confusion_matrix.max() / 2  # color threshold
    plt.figure(figsize=(10, 10))
    plt.title("Confusion Matrix")
    for x in range(num_class):  # actual labels
        for y in range(num_class):  # predicted labels
            info = int(confusion_matrix[y, x])
            plt.text(x, y, info, verticalalignment='center', horizontalalignment='center',
                     color='white' if info > thresh else 'black')
    plt.tight_layout()
    plt.yticks(range(num_class), labels=class_names)
    plt.xticks(range(num_class), labels=class_names, rotation=45)
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.savefig(plot_file, format='png', bbox_inches='tight', dpi=300, transparent=True)
    # plt.show()
    plt.close()
    print("Output confusion matrix => {}".format(str(out_path)))
    return precision, recall

def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def init_seeds(seed):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters?
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y

    # Convert keypoints from [x, y, v] normalized to [x, y]
    if y.shape[-1] > 4:
        nl = y.shape[0]
        kp = y[:, 4:].reshape(nl, -1, 3)
        kp[..., 0] *= w
        kp[..., 0] += padw
        kp[..., 1] *= h
        kp[..., 1] += padh
        y[:, 4:] = kp.reshape(nl, -1)

    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height

    # convert keypoints from [x, y, v] to [x, y, v] normalized
    if y.shape[-1] > 4:
        nl = y.shape[0]
        kp = y[:, 4:].reshape(nl, -1, 3)
        kp[..., 0] /= w
        kp[..., 1] /= h
        y[:, 4:] = kp.reshape(nl, -1)

    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2