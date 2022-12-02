import torch
# from pose.kapao_with_kp_conf.models.experimental import attempt_load
from pathlib import Path

device = torch.device('cpu')
model = torch.load("runs/cls_128/cls_img/weights/best.pt",
                   map_location=device)  # load FP32 model

# print(model.keys())
model = model['ema'].float().eval()

x1 = torch.rand(1, 3, 128, 128)  # img
x2 = torch.rand(1, 16, 26)   # sequences
trace_model = torch.jit.trace(model, (x1, x2))
save_path = Path("./jit_model.pt")
torch.jit.save(trace_model, save_path)
print(f"{save_path.absolute()}")
