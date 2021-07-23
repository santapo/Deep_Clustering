from torch import optim
import torch.nn as nn
from autoencoder import AutoEncoder

import pytorch_lightning as pl


class AutoEncoderModel(pl.LightningModule):
    def __init__(self,
                optimizer_name: str = "Adam",
                lr: float = 0.001,
                pretrained: bool = True,
                freeze_encoder: bool = False):
        """AutoEncoderModel
        Args:
        
        """
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.inp_freeze_encoder = freeze_encoder
        self.model = AutoEncoder(pretrained=pretrained)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        """
        Model Forward Path
        """
        return self.model(x)
    
    def configure_optimizers(self):
        """
        Configuration of Optimizer and LR Scheduler
        """
        if self.inp_freeze_encoder:
            trainable_parameters = self.model.decoder.parameters()
        else:
            trainable_parameters = self.model.parameters()
        
        if self.optimizer_name == "Adam":
            optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer_name} is not defined')
        
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch of an training iteration.
        """
        image = batch
        out = self.forward(image)
        loss = self.criterion(out, image)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch of a validation iteration.
        """
        image = batch
        out = self.forward(image)
        loss = self.criterion(out, image)
        self.log('val_loss', loss, on_step=True)

    

    