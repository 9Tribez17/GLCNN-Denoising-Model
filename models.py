import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


class GMC_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_train, noise,imgn_train):
        l2_loss = torch.sum(torch.pow((out_train - noise), 2)) / (imgn_train.size()[0]*2)

        h_x = out_train.size()[2]
        w_x = out_train.size()[3]
        count_h = self._tensor_size(out_train[:, :, 1:, :])
        count_w = self._tensor_size(out_train[:, :, :, 1:])
        h_tv = torch.pow((out_train[:, :, 1:, :] - out_train[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((out_train[:, :, :, 1:] - out_train[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        h_x1 = noise.size()[2]
        w_x1 = noise.size()[3]
        count_h1 = self._tensor_size(noise[:, :, 1:, :])
        count_w1 = self._tensor_size(noise[:, :, :, 1:])
        h_tv1 = torch.pow((noise[:, :, 1:, :] - noise[:, :, :h_x - 1, :]), 2).sum()
        w_tv1 = torch.pow((noise[:, :, :, 1:] - noise[:, :, :, :w_x - 1]), 2).sum()
        tvloss1 = h_tv1 / count_h1 + w_tv1 / count_w1

        gmc_loss = 0.001 * abs(tvloss-tvloss1) + l2_loss

        return gmc_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))

class PGMC_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_train, noise,imgn_train):
        h_x = out_train.size()[2]
        w_x = out_train.size()[3]
        count_h = self._tensor_size(out_train[:, :, 1:, :])
        count_w = self._tensor_size(out_train[:, :, :, 1:])
        h_gmc = torch.pow((out_train[:, :, 1:, :] - out_train[:, :, :h_x - 1, :]), 2).sum()
        w_gmc = torch.pow((out_train[:, :, :, 1:] - out_train[:, :, :, :w_x - 1]), 2).sum()
        gmcloss = h_gmc / count_h + w_gmc / count_w

        SmoothL1Loss = self.smooth_l1_loss(out_train,noise,imgn_train,reduce=True)/ (imgn_train.size()[0]*2)
        pgmc_loss = 0.01 * gmcloss + SmoothL1Loss
        return pgmc_loss

    def smooth_l1_loss(self,input, target,imgn_train,reduce=True):
        diff = torch.abs(input - target);cond = diff < 1; loss = torch.where(cond, 0.5 * diff ** 2, diff - 0.5)
        if reduce:
            return torch.sum(loss)
        return torch.sum(loss, dim=1)

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]