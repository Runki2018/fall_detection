import argparse
import onnx
from pathlib import Path
import torch
import torch.onnx
import yaml
import sys
# FILE = Path(__file__).absolute().parents[1].as_posix()
# print(f"{FILE=}")
# sys.path.append(FILE)  # add yolov5/ to path
# sys.path.pop(0)
print(sys.path)

from models.experimental import attempt_load
from models.yolo import Detect, Model
# from utils.datasets import LoadImages
from utils.general import (Profile, check_dataset, check_img_size,
                           check_requirements, check_version, colorstr, file_size)
from utils.torch_utils import select_device

def export_onnx(opt):
    check_requirements('onnx')
    print(f"ONNX: starting export with onnx {onnx.__version__}...")

    # 1. Load model
    data_path = Path(opt.data)
    print(f"{data_path.exists()=}")
    data_dict = check_dataset(opt.data, autodownload=False)  # check if None
    num_coords = data_dict.get('num_coords', 0)  # 34 for COCO
    nc = int(data_dict['nc'])  # number of classes
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    device = select_device(opt.device)
    ckpt = torch.load(opt.weights, map_location=device)
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3,
                  nc=nc, anchors=hyp.get('anchors'), num_coords=num_coords)
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    model.load_state_dict(csd, strict=False)
    model.eval()
    model = model.to(device)

    # 2. ONNX setting
    onnx_file = Path(opt.weights).with_suffix('.onnx')
    output_names = ['output0']
    if opt.dynamic:
        dynamic = {'image': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    input_img = torch.rand(1, 3, opt.img_size, opt.img_size).to(device)

    # 3. Export ONNX
    torch.onnx.export(
        model.cpu() if dynamic else model,   # --dynamic only compatible with cpu
        input_img.cpu() if dynamic else input_img,
        onnx_file,   # output file
        verbose=False,
        opset_version=opt.opset,  # ONNX operator version
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # 4. Check
    model_onnx = onnx.load(onnx_file)
    onnx.checker.check_model(model_onnx)

    # 5. Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, onnx_file)

    # 6. Simplify
    if opt.simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            print(f'ONNX: simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'ONNX: simplifier failure: {e}')

    print(f"=> save ONNX file to {onnx_file}.")
    return onnx_file, model_onnx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/v7tiny_e300_640/dm100_wm522/weights/best.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.kp.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    opt = parser.parse_args()
    print(colorstr('ONNX: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    export_onnx(opt)



