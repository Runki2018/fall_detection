import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
import torch.multiprocessing as mp
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.classifer import Classifer, ComputeLoss
from utils.video_dataloader import create_dataloader, create_cache

from kapao_utils.plots import plot_labels, plot_evolve, plot_lr_scheduler
from kapao_utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, \
    select_device, torch_distributed_zero_first, init_torch_seeds
from kapao_utils.general import one_cycle, init_seeds, check_file, colorstr, get_latest_run, \
    increment_path, strip_optimizer, plot_confusion_matrix, fitness
from tensorboardX import SummaryWriter


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # project output directory
    parser.add_argument('--project', default='runs/train_debug', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyp_cls.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--port', type=str, default='23456', help='DDP init method, change it when Address conflict')
    # parser.add_argument('--init_method', default='tcp://127.0.0.1:12345', help='DDP init method')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--search-cfg', action='store_true', help='search the best depth and width multiple')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')

    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # Don't change, auto value  https://pytorch.org/docs/stable/elastic/run.html#launcher-api
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train(opt, device, RANK, WORLD_SIZE):
    RANK = int(RANK)  # RANK must be int type, otherwise being blocked in torch_distributed_zero_first
    opt.save_dir = Path(opt.save_dir)
    # Directories
    w = opt.save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    # Save run settings
    with open(opt.save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        opt.save_dir = str(opt.save_dir)
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        opt.save_dir = Path(opt.save_dir)
    os.system(f"cp ./dist_train_run.sh {str(opt.save_dir / 'dist_train_run.sh')}")
    data_dict = None

    # Config
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)

    data_root = hyp['data_root']
    with torch_distributed_zero_first(RANK):
        train_videos, test_videos = create_cache(data_root)
    num_class = hyp['num_class']  # number of classes
    class_names = hyp['names']  # class name
    assert len(class_names) == num_class, f'{len(class_names)} names found for {num_class=} dataset in {opt.hyp}'

    # Model
    model = Classifer(num_class=num_class,
                      seq_num=hyp['seq_num'],
                      x_size=hyp['seq_features'],
                      y_size=hyp['proj_size'],
                      num_layers=hyp['num_layers'],
                      dropout=hyp['dropout'],
                      min_hidden_size=hyp['min_hidden_size'],
                      mid_c=hyp['mid_c'])
    criterion = ComputeLoss(device=device, weight=hyp['weight'], pos_weight=hyp['pos_weight'])
    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model.load_state_dict(csd, strict=False)

    # Trainloader
    # with torch_distributed_zero_first(RANK):
    train_loader, _ = create_dataloader(opt.batch_size, opt.workers, train_videos, opt.imgsz,
                                        hyp['seq_num'], hyp['seq_interval'], is_train=True,
                                        train_multi=hyp['train_multi'],
                                        prob_seq_zero=hyp['prob_seq_zero'], prob_kp_zero=hyp['prob_kp_zero'])

    nb = len(train_loader)  # number of batches
    # Process 0
    if RANK in [-1, 0]:
        writer = SummaryWriter(logdir=str(opt.save_dir))
        val_loader, _ = create_dataloader(opt.batch_size, opt.workers, train_videos,
                                       opt.imgsz, hyp['seq_num'], hyp['seq_interval'], is_train=False)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP mode
    model = model.to(device)
    model = DDP(model, device_ids=[RANK], output_device=RANK)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust betal to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], opt.epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, opt.epochs, save_dir=opt.save_dir)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, f'{opt.weights} training to {opt.epochs} epochs is finished, nothing to resume.'
        if opt.epochs < start_epoch:
            opt.epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd

    # Start training
    num_warmup = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, opt.epochs):  # epoch --------------------------------------------------------------
        model.train()
        mloss = torch.zeros(1, device=device)  # mean loss
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (images, seq_features, targets) in pbar:  # mini-batch iteration ----------------
            ni = i + nb * epoch  # number integrated batches (since train start)

            # Warmup
            if ni <= num_warmup:
                xi = [0, num_warmup]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            images = images.to(device, non_blocking=True)  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(images, seq_features)  # forward
                loss, loss_items = criterion(pred, targets)
                # loss, loss_items = model.compute_loss(pred, targets)
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
            # Backward
            # with torch.autograd.detect_anomaly():
            scaler.scale(loss).backward()

            # Optimize after warmup
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 3) %
                                     (f'{epoch}/{opt.epochs - 1}', mem, *mloss, targets.shape[0], images.shape[-1]))
            # end batch -------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups[:3]]  # for loggers
        scheduler.step()

        stop = False
        # with torch_distributed_zero_first(RANK):
        if RANK in [-1, 0]:
            # mAP
            writer.add_scalars("lr", {f"lr_g{i}": lr_i for i, lr_i in enumerate(lr)}, epoch)
            writer.add_scalar("train_loss", mloss.item(), epoch)
            mloss = torch.zeros(1, device=device)  # mean loss for validation
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == opt.epochs) or stopper.possible_stop

            with torch.no_grad():
                model.eval()
                confusion_matrix = torch.zeros((num_class, num_class))
                pbar = tqdm(enumerate(val_loader), desc="validating", total=len(val_loader))
                for i, (images, seq_features, targets) in pbar:  # mini-batch iteration ----------------
                    preds = model(images.to(device), seq_features)  # forward
                    loss, loss_items = criterion(pred, targets)
                    # loss, loss_items = model.compute_loss(pred, targets)
                    targets = targets.squeeze().to(device)
                    preds = torch.argmax(preds, dim=1)
                    for p, t in zip(preds.type(torch.long), targets.type(torch.long)):
                        confusion_matrix[p, t] += 1

                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                results = plot_confusion_matrix(confusion_matrix, opt.save_dir, class_names)
                writer.add_scalar("val_loss", mloss.item(), epoch)
            # Update best mAP
            fi = fitness(results, b=hyp['beta'])  # weighted combination of [P, R]
            writer.add_scalars('precision', {f"p{k}": p for k, p in enumerate(results[0])}, epoch)
            writer.add_scalars('recall', {f"r{k}": r for k, r in enumerate(results[1])}, epoch)
            writer.add_scalar(f"F1_beta({hyp['beta']}", fi, epoch)
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,  # 0.1*AP50 + 0.9*mAP
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
            stop = stopper(epoch=epoch, fitness=fi)  # get stop flag on Rank 0
        # dist.barrier(device_ids=[RANK])

        # check_rank_block(RANK, prefix='before_early_stop')
        # Early Stop  https://github.com/ultralytics/yolov5/pull/4576
        broadcast_list = [stop] if RANK == 0 else [None]
        dist.broadcast_object_list(broadcast_list, src=0)  # broadcast 'stop' to all ranks
        stop = broadcast_list[0]
        if stop:  # Stop Single GPU and Multi GPU training
            break
        # end epoch ------------------------------------------------------------------------------------------

    # end training -------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        if not opt.evolve:
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
    torch.cuda.empty_cache()
    return results


def check_rank_block(RANK, prefix=''):
    # if a rank does not been blocked, it will save a file '*.rank' on the root directory.
    file = Path(prefix + '_' + str(RANK) + '.rank')
    with file.open('w') as fd:
        fd.write(str(RANK))

def main(RANK, opt, WORLD_SIZE):
    # LOCAL_RANK = RANK  # single machine with multi GPUs
    # Checks

    if RANK == 0:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    master_port = opt.port

    # Resume
    if opt.resume and not opt.evolve:
        # specified or most recent path
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'

        epochs, device, batch, workers = opt.epochs, opt.device, opt.batch_size, opt.workers
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace !!!!!!
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        opt.epochs, opt.device, opt.batch_size, opt.workers = epochs, device, batch, workers

        print(f'Resuming training from {ckpt}')
    else:
        opt.hyp = check_file(opt.hyp)  # check files
        # assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
    assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
    # assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
    assert not opt.evolve, '--evolve argument is not compatible with DDP training'
    torch.cuda.set_device(RANK)
    device = torch.device('cuda', RANK)
    DDP_setup(RANK, WORLD_SIZE, master_port)

    # Train
    if not opt.evolve and not opt.search_cfg:
        train(opt, device, RANK, WORLD_SIZE)
    # Evolve hyperparameters (optional)
    elif opt.evolve:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv from Google Cloud if exists
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0 ~ 1
                ng = len(meta)
                v = np.ones(ng)
                mp, s = 0.8, 0.2  # mutation probability, sigma
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
            # Train mutation
            results = train(hyp.copy(), opt, device)
            # Write mutation results
            # print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')
    elif opt.search_cfg:  # search the best depth and width within model size 10M
        import json
        save_dir = Path(opt.save_dir)
        model_size_file = str(save_dir / 'model_size.json')
        if not save_dir.exists():
            save_dir.mkdir()
        if os.path.exists(model_size_file):
            with open(model_size_file, 'r') as fd:
                size_dict = json.load(fd)
        else:
            size_dict = dict(best_ap=0, best_iter=0, current_iter=0, results=[], multiples=[], model_size=[])
            with open(opt.cfg, 'r') as fd:
                yaml_dict = yaml.safe_load(fd)
            max_dm, max_wm = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
            min_dm, min_wm = 0.5, 0.5
            min_size, max_size, step_size = 3, 12, 0.02

            dm = max_dm
            while dm >= min_dm:
                wm = max_wm
                while wm >= min_wm:
                    yaml_dict['depth_multiple'], yaml_dict['width_multiple'] = dm, wm
                    model = Classifer()  # TODO: this
                    wm -= step_size
                    if model.sum_params in size_dict['model_size'] or \
                        model.sum_params > max_size or model.sum_params < min_size:  # 3~12M
                        del model
                        continue
                    else:
                        size_dict['multiples'].append((round(dm, 2), round(wm, 2)))
                        size_dict['model_size'].append(model.sum_params)
                        del model
                dm -= step_size
            if RANK == 0 :
                with open(model_size_file, 'w') as fd:
                    json.dump(size_dict, fd, indent=4)

        with open(opt.cfg, 'r') as fd:
            yaml_dict = yaml.safe_load(fd)
        iter = size_dict['current_iter']
        num_size = len(size_dict['model_size'])
        while iter < num_size:
            # change dm, wm
            yaml_dict['depth_multiple'] = size_dict['multiples'][iter][0]
            yaml_dict['width_multiple'] = size_dict['multiples'][iter][1]
            opt.cfg = yaml_dict
            results = train(opt, device, RANK, WORLD_SIZE)

            if RANK ==0:
                # record results
                size_dict['current_iter'] = iter
                results = [round(v, 5) for v in results[:4]]
                size_dict['results'].append(results)  # mp, mr, mAP50, mAP
                if results[3] > size_dict['best_ap']:
                    size_dict['best_ap'] = results[3]
                    size_dict['best_iter'] = iter
                with open(model_size_file, 'w') as fd:
                    json.dump(size_dict, fd, indent=4)
                print(f"=> Searching {iter+1}/{num_size}...\n"
                    f"=> Best mAP = {size_dict['best_ap']} with "
                    f"(depth_multiple, width_multiple) = {size_dict['multiples'][size_dict['best_iter']]}.\n"
                    f"=> Model size = {size_dict['model_size'][size_dict['best_iter']]}")
            iter += 1
    DDP_cleanup(RANK)

# ----------------Don't Modify-------------------------------------------------
def DDP_setup(rank, world_size, port='12345'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                            rank=rank, world_size=world_size)

def DDP_cleanup(RANK=0):
    if RANK == 0:
        print('Destroying process group ...', end='')
        dist.destroy_process_group()
        print('Done.')

if __name__ == "__main__":
    opt = parse_opt()
    num_gpus = torch.cuda.device_count()
    device_list = opt.device.strip(' ').split(',')
    if '' in device_list:
        device_list.remove('')
    if 0 < len(device_list) < num_gpus:
        # assert int(device_list[0]) < num_gpus, "device id should be [0,num_gpus-1]!"
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
        num_gpus = len(device_list)

    try:
        mp.spawn(
            main,
            args=(opt, num_gpus),
            nprocs=num_gpus,
            join=True
        )
    except KeyboardInterrupt:
        DDP_cleanup()
    finally:
        torch.cuda.empty_cache()
        # os.system('rm ./*.rank')
