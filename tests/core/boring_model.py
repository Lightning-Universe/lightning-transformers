import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lightning_transformers.core import LitTransformer


class BoringDataModule(LightningDataModule):

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringTransformerModel(LitTransformer):

    def __init__(self):
        super().__init__(model=torch.nn.Linear(32, 2))

    def forward(self, x):
        return self.model(x)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
