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

import val
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, \
    init_seeds, strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve, plot_lr_scheduler
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, \
    select_device, torch_distributed_zero_first
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # project output directory
    parser.add_argument('--project', default='runs/train_debug', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/pose12-kp.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.kp.yaml', help='hyperparameters path')
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
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # parser.add_argument('--autobalance', action='store_true', help='Learn keypoint and object loss scaling (experimental)')

    # evaluate
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--val-scales', type=float, nargs='+', default=[1])
    parser.add_argument('--val-flips', type=int, nargs='+', default=[-1])

    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')

    # Don't change, auto value  https://pytorch.org/docs/stable/elastic/run.html#launcher-api
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train(opt, device, RANK, WORLD_SIZE, callbacks=Callbacks()):
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
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(opt.save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        opt.save_dir = str(opt.save_dir)
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        opt.save_dir = Path(opt.save_dir)
    os.system(f"cp ./dist_train_run.sh {str(opt.save_dir / 'dist_train_run.sh')}")
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(opt.save_dir, opt.weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(opt.data)  # check if None
    # check_rank_block(RANK, prefix="check_dataset")
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class name
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {opt.data}'

    labels_dir = data_dict.get('labels', 'labels')
    kp_flip = data_dict.get('kp_flip')
    kp_bbox = data_dict.get('kp_bbox')
    num_coords = data_dict.get('num_coords', 0)

    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        # check_rank_block(RANK, prefix="attempt_download")
        ckpt = torch.load(weights, map_location=device)
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc,
                      anchors=hyp.get('anchors'), num_coords=num_coords)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), num_coords=num_coords).to(device)

    # Freeze 
    freeze = [f'model.{x}.' for x in range(opt.freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Image sizes
    grid_size = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, grid_size, floor=grid_size * 2)  # verify imgsz is gs-multiple

    # Trainloader
    # with torch_distributed_zero_first(RANK):
    train_loader, dataset = create_dataloader(
        train_path, labels_dir, imgsz, opt.batch_size // WORLD_SIZE, grid_size,opt.single_cls,hyp=hyp, augment=True,
        cache=opt.cache, rect=opt.rect, rank=RANK, workers=opt.workers, image_weights=opt.image_weights,
        quad=opt.quad, prefix=colorstr('train: '), kp_flip=kp_flip, kp_bbox=kp_bbox)
    # check_rank_block(RANK, prefix="create_dataloader")

    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            val_path, labels_dir, imgsz, opt.batch_size // WORLD_SIZE, grid_size, opt.single_cls, hyp=hyp,
            cache=None if opt.noval else opt.cache, rect=False, rank=-1, workers=opt.workers, pad=0.5,
            prefix=colorstr('val: '), kp_flip=kp_flip, kp_bbox=kp_bbox)[0]

        if not opt.resume:
            # labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            # if plots:
            #     plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
        callbacks.on_pretrain_routine_end()

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['kp'] *= 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names  # class names
    compute_loss = ComputeLoss(model, autobalance=False, num_coords=num_coords,
                               device=device, joint_weights=data_dict.get('joint_weights', None))  # init loss class

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        LOGGER.info('Using SyncBatchNorm()')

    # DDP mode
    model = model.to(device)
    model = DDP(model, device_ids=[RANK], output_device=RANK)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

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
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
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
            assert start_epoch > 0, f'{weights} training to {opt.epochs} epochs is finished, nothing to resume.'
        if opt.epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs."
                        f" Fine-tuning for {opt.epochs} more epochs.")
            opt.epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd

    # Start training
    t0 = time.time()
    num_warmup = max(round(hyp['warmup_epochs'] * nb),
                     1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls, kp)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', opt.save_dir)}\n"
                f'Starting training for {opt.epochs} epochs...')

    for epoch in range(start_epoch, opt.epochs):  # epoch --------------------------------------------------------------
        model.train()
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean loss
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 8) % \
                    ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'kps', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (images, targets, paths, _) in pbar:  # mini-batch iteration ----------------
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

            images = images.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # Multi-scale
            if opt.multi_scale:
                size = random.randrange(imgsz * 0.5, imgsz * 1.5) // grid_size * grid_size
                scale_factor = size / max(images.shape[2:])
                if scale_factor != 1:
                    # new shape (stretched to gs-multiple)
                    new_shape = [math.ceil(x * scale_factor / grid_size) * grid_size for x in images.shape[2:]]
                    images = nn.functional.interpolate(images, size=new_shape, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(images)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
            # Backward
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
                pbar.set_description(('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{opt.epochs - 1}', mem, *mloss, targets.shape[0], images.shape[-1]))
                callbacks.on_train_batch_end(ni, model, images, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups[:3]]  # for loggers
        scheduler.step()

        stop = False
        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == opt.epochs) or stopper.possible_stop
            if not opt.noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                           batch_size=opt.batch_size // WORLD_SIZE,
                                           imgsz=imgsz,
                                           conf_thres=0.01,
                                           model=ema.ema,
                                           dataloader=val_loader,
                                           compute_loss=compute_loss,
                                           scales=opt.val_scales,
                                           flips=[None if f == -1 else f for f in opt.val_flips])
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,  # 0.1*AP50 + 0.9*mAP
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)
            stop = stopper(epoch=epoch, fitness=fi)  # get stop flag on Rank 0

        # check_rank_block(RANK, prefix='before_early_stop')
        # Early Stop  https://github.com/ultralytics/yolov5/pull/4576
        broadcast_list = [stop] if RANK == 0 else [None]
        dist.broadcast_object_list(broadcast_list, src=0)  # broadcast 'stop' to all ranks
        stop = broadcast_list[0]
        if stop:  # Stop Single GPU and Multi GPU training
            break
        # check_rank_block(RANK, prefix='after_early_stop')
        # end epoch ------------------------------------------------------------------------------------------

    # end training -------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not opt.evolve:
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', opt.save_dir)}")

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
    set_logging(RANK)
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

        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    # device = select_device(opt.device, batch_size=opt.batch_size)
    # from datetime import timedelta
    assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
    assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
    assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
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
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)
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
                    model = Model(cfg=yaml_dict, nc=18)
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
                LOGGER.info(f"=> Searching {iter+1}/{num_size}...\n"
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
