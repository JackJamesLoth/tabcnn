import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import os

from .datasets.tabCNN_guitarset import TabCNN_GuitarSet

class TabCNNData(pl.LightningDataModule):
    def __init__(self,
                 data_path, 
                 batch_size=128, 
                 shuffle=True,
                 spec_repr="c", 
                 con_win_size=9,
                 val_split=0.2, 
                 test_split=0.1,
                 num_workers=1
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None):

        # Create dataset
        file_names = [f for f in os.listdir(self.hparams.data_path + self.hparams.spec_repr) if f.endswith('.npz')]
        dataset = TabCNN_GuitarSet(file_names, self.hparams.data_path, self.hparams.spec_repr, self.hparams.con_win_size)

        # Splitting dataset
        val_size = int(len(dataset) * self.hparams.val_split)
        test_size = int(len(dataset) * self.hparams.test_split)
        train_size = len(dataset) - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)