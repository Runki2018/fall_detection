import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from utils.video_dataloader import create_dataloader, create_cache
from kapao_utils.general import plot_confusion_matrix, fitness


hyp = "hyp_cls.yaml"
with open(hyp) as f:
    hyp = yaml.safe_load(f)  # load hyps dict

train_videos, test_videos = create_cache(hyp['data_root'])
val_loader, _ = create_dataloader(8, 2, train_videos, 128,
                                  hyp['seq_num'], hyp['seq_interval'],
                                  is_train=False, dist=False)
num_class = 3
device = torch.device('cpu')
model = torch.jit.load("jit_model.pt", map_location=device)  # load FP32 model


with torch.no_grad():
    model.eval()
    confusion_matrix = torch.zeros((num_class, num_class))
    pbar = tqdm(enumerate(val_loader), desc="validating", total=len(val_loader))
    for i, (images, seq_features, targets) in pbar:  # mini-batch iteration ----------------
        preds = model(images.to(device), seq_features.to(device))  # forward
        # loss, loss_items = criterion(pred, targets)
        # loss, loss_items = model.compute_loss(pred, targets)
        targets = targets.squeeze().to(device)
        preds = torch.argmax(preds, dim=1)
        for p, t in zip(preds.type(torch.long), targets.type(torch.long)):
            confusion_matrix[p, t] += 1

        # mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
    save_dir = Path('.')
    results = plot_confusion_matrix(confusion_matrix, save_dir, hyp['names'])