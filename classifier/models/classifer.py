import torch
from torch import nn

class ComputeLoss(nn.Module):
    def __init__(self, device, weight=(0.2, 1, 0.8), pos_weight=(0.5, 1, 1)):
        super(ComputeLoss, self).__init__()
        # loss
        # self.criterion = nn.BCELoss(weight=torch.tensor(weight, device=device))
        self.criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(weight, device=device),
                                              pos_weight=torch.tensor(pos_weight, device=device))

    def forward(self, p, labels):
        cols = labels.type(torch.long)
        rows = torch.arange(0, p.shape[0])
        y = torch.zeros_like(p)
        y[rows, cols] = 1
        loss = self.criterion(p, y.to(p.device))
        return loss, loss.detach()


class Classifer(nn.Module):
    def __init__(self, num_class=3, seq_num=16, x_size=37, y_size=None, num_layers=2,
                 dropout=0.2, mid_c=64, min_hidden_size=128):
        super().__init__()
        # input [Batch, seq_num, x_size] -> output [Batch, seq_num, y_size]
        y_size = y_size if y_size is not None else x_size
        self.lstm = nn.LSTM(input_size=x_size,
                            hidden_size=max(x_size*4, min_hidden_size),
                            proj_size=y_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)

        self.seq_attention = nn.Sequential(
            nn.LayerNorm([seq_num, y_size]),
            nn.Linear(y_size, y_size),
            # nn.BatchNorm1d(seq_num),
            nn.LayerNorm([seq_num, y_size]),
            nn.ReLU(),
            nn.Linear(y_size, 1),
            nn.Sigmoid()
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((9, 9))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((9, 9))
        self.conv = nn.Sequential(
            nn.Conv2d(6, mid_c, 1, 1, 0),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, mid_c // 2, 1, 1, 0),
            nn.BatchNorm2d(mid_c // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(mid_c // 2 * 81, y_size)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(2*y_size, min_hidden_size),
            nn.LayerNorm([min_hidden_size]),
            # nn.BatchNorm1d(min_hidden_size),
            nn.ReLU(),
            nn.Linear(min_hidden_size, num_class),
            # nn.Sigmoid()
        )

    def forward(self, img, seq):
        # self.assert_tensor(img)
        # self.assert_tensor(seq)
        y_seq = self.lstm(seq)[0]
        # self.assert_tensor(y_seq)
        y_seq = y_seq * self.seq_attention(y_seq)
        # self.assert_tensor(y_seq)
        y_seq = torch.sum(y_seq, dim=1)
        # self.assert_tensor(y_seq)
        y_img1 = self.adaptive_max_pool(img)
        # self.assert_tensor(y_img1)
        y_img2 = self.adaptive_avg_pool(img)
        # self.assert_tensor(y_img2)
        y_img = self.conv(torch.cat([y_img1, y_img2], dim=1))
        # self.assert_tensor(y_img)
        y = torch.cat([y_seq, y_img], dim=1)
        # self.assert_tensor(y)
        y = self.out_layer(y)
        # self.assert_tensor(y)
        return y

    def assert_tensor(self, t):
        assert not torch.any(torch.isnan(t))
        assert not torch.any(torch.isinf(t))


if __name__ == '__main__':
    x_size = int(12*3 + 1)  # 2*num_joints + wh_ratio
    len_seq = 16  # 16 frame
    model = Classifer(num_class=3,
                      seq_num=len_seq,
                      x_size=x_size,
                      y_size=64,
                      num_layers=2,
                      dropout=0.2,
                      mid_c=64,
                      min_hidden_size=128)

    img = torch.rand(1, 3, 64, 64)
    kpt_seq = torch.rand(1, len_seq, x_size)
    y = model(img, kpt_seq)
    print(f"{y.shape=}")
    print(f"{y=}")

    labels = torch.randint(0, 2, size=(1, 16))
    loss = model.compute_loss(y, labels)
    print(f"{loss=}")

