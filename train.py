import torch
from torch.nn import functional as F

from opt import get_opts

# datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

# models
from models.networks import LinearModel

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(1234, workers=True)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MNISTSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = LinearModel(self.hparams.hidden_dim)
        
    def forward(self, x):
        return self.net(x)
        
    def prepare_data(self):
        """
        download data once
        """
        MNIST(self.hparams.root_dir, train=True, download=True)
        MNIST(self.hparams.root_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        dataset = MNIST(self.hparams.root_dir,
                        train=True,
                        download=False,
                        transform=T.ToTensor())
        train_length = len(dataset) # 60000
        self.train_dataset, self.val_dataset = \
            random_split(dataset,
                         [train_length-self.hparams.val_size, self.hparams.val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.hparams.num_epochs,
                                      eta_min=self.hparams.lr/1e2)

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits_predicted = self(images)

        loss = F.cross_entropy(logits_predicted, labels)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits_predicted = self(images)

        loss = F.cross_entropy(logits_predicted, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits_predicted, -1), labels).to(torch.float32)) / len(labels)

        log = {'val_loss': loss,
               'val_acc': acc}

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/acc', mean_acc, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    mnistsystem = MNISTSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/acc',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=1,
                      benchmark=True)

    trainer.fit(mnistsystem)