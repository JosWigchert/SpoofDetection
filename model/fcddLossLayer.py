import torch
import torch.nn as nn


class FcddLossLayer(nn.Module):
    def __init__(self):
        super(FcddLossLayer, self).__init__()

    def forward(self, Y, T):
        normal_term = Y
        anomaly_term = torch.log(1 - torch.exp(-normal_term))

        is_good = ~T
        loss = torch.mean(
            is_good.float() * normal_term - ~is_good.float() * anomaly_term
        )
        return loss
