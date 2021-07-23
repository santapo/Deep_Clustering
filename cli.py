import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from pytorch_lightning.callbacks import ModelCheckpoint

from data import DeepClusteringDataModule
from model import AutoEncoderModel

class AutoEncoderTrainingCLI(LightningCLI):
    """
    CLI for training AutoEncoder model
    """
    def add_callbacks(self):
        save_checkpoint = ModelCheckpoint(dirpath=None,
                                          filename='{epoch}-{step}-{validation_loss:.2f}-{validation_F1:.2f}',
                                          save_top_k=-1)
        self.config_init['trainer']['callbacks'] = [save_checkpoint]

    def instantiate_trainer(self):
        self.add_callbacks()
        super().instantiate_trainer()

    # def instantiate_model(self):
    #     temp_train_loader = self.datamodule.train_dataloader()
    #     train_loader_total_samples = len(temp_train_loader)
    #     import ipdb; ipdb.set_trace()
    #     self.config_init['model']['train_loader_total_samples'] = train_loader_total_samples
    #     super().instantiate_model()

def cli_main():
    AutoEncoderTrainingCLI(AutoEncoderModel, DeepClusteringDataModule,
                            seed_everything_default=42)

if __name__ == "__main__":
    cli_main()
