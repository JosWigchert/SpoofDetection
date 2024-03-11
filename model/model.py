import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.models import vgg16
import lightning as L

from model import FcddLossLayer

# define any number of nn.Modules (or use your current ones)


# define the LightningModule
class LitFCDD(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.vgg16 = vgg16()
        self.fcddLossLayer = FcddLossLayer()

        # Define additional layers
        self.additionalFCLayers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.FunctionalLayer(lambda x: torch.sqrt(x**2 + 1) - 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            self.fcddLossLayer,
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
