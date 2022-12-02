import cv2
import numpy as np
import onnxruntime
import torch
# import torch.onnx
import onnx
from torch import nn
import cv2
from models.yolo import Model as KAPAO
from pathlib import Path

class Model(nn.Module):
    def __init__(self, upscale_factor=2, cat_dim=1):
        super(Model, self).__init__()
        self.upscale_factor = upscale_factor
        self.cat_dim = cat_dim
        self.up = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='nearest')

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 32, 5, 1, 2)

    def forward(self, x):
        x = self.up(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        out = torch.cat([y1, y2], self.cat_dim)
        return out


if __name__ == '__main__':
    export_kapao = True
    dynamic = False
    opencv_check = True

    # load and check model
    img = torch.rand((1, 3, 640, 640))
    if export_kapao:
        model = KAPAO(cfg="/home/huangzhiyong/Project/kapao/models/yolov7-tiny-pose.yaml",
                  nc=18, num_coords=34)
        ckt = torch.load('../runs/v7tiny_e300_640/dm100_wm52_r90_e60_finetune5/weights/best.pt')
        model.load_state_dict(ckt['model'].float().state_dict(), strict=False)
        # model = torch.jit.script(model)
    else:
        model = Model(upscale_factor=2, cat_dim=1)
    model = model.eval()
    print(f"EVAl : {model.training=}")

    y = model(img)
    if isinstance(y, tuple):
        for i, yi in enumerate(y):
            if isinstance(yi, (list, tuple)):
                for yii in yi:
                    print(f"{i}: {yii.shape=}")
            else:
                print(f"{i}: {yi.shape=}")
    else:
        print(f"{y.shape=}")

    # export onnx
    output_onnx = "./model.onnx"
    with torch.no_grad():
        if export_kapao:
            torch.onnx.export(
                model, img, output_onnx,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 1: 'anchors'}
                } if dynamic else None
            )
        else:
            torch.onnx.export(
                model,
                img,
                output_onnx,
                opset_version=11,
                input_names=['images'],
                output_names=['output'])

    # check onnx
    model_onnx = onnx.load(output_onnx)
    try:
        onnx.checker.check_model(model_onnx)
    except Exception:
        print("ONNX: Model incorrect")
    else:
        print(f"ONNX: Model correct => {Path(output_onnx).absolute()}")

    # Inference Engine
    if opencv_check: # opencv load onnx
        image = cv2.imread("/home/huangzhiyong/Project/kapao/res/crowdpose_100024.jpg")
        image = cv2.resize(image, (640, 640))
        model_cv = cv2.dnn.readNetFromONNX(output_onnx)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, swapRB=True)
        print(f"{type(blob)=}\t{blob.shape=}")
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        # 1. Set the input to the network
        model_cv.setInput(blob, 'images')

        # 2. Runs the forward pass to get output of the output layers
        # TODO: check the dimensions of outs [na*(w1*h1+w2*h2+w3*h3), 6 + 3*num_kp]
        outs1 = model_cv.forward()
        print(f"{len(outs1)=}")
        print(f"{outs1.shape=}")
        # outs2 = model_cv.forward(model_cv.getUnconnectedOutLayersNames())  # get all output
        # print(f"{len(outs2)=}")
        # print(f"{outs2[0].shape=}")
        # print(f"{outs2[0][0]=}")
        # print((f"{model(torch.tensor(blob))[0]=}"))
        outs2 = model(torch.tensor(blob)).detach().numpy()
        dist = np.linalg.norm(outs1-outs2, axis=-1).mean()
        print(f"MSE[Model output - CV output] = {dist}")
    else:
        session = onnxruntime.InferenceSession(output_onnx)
        session.get_modelmeta()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"session: {input_name=}\t{output_name=}")
        inputs = torch.randn((1, 3, 640, 640), dtype=torch.float32).numpy()
        outputs = session.run([output_name], {input_name: inputs})
        print(outputs)