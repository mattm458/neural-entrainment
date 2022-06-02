import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class FixedHistoryEntrainmentModel(pl.LightningModule):
    def __init__(
        self, lr=0.0001, rnn_in_dim=3, rnn_out_dim=32, out_dim=3, has_attention=False
    ):
        super().__init__()

        if has_attention:
            raise NotImplementedError("Attention not implemented")

        self.lr = lr

        self.rnn = nn.LSTM(rnn_in_dim, rnn_out_dim, batch_first=True)
        self.linear = torch.nn.Linear(rnn_out_dim, out_dim)

    def forward(self, X):
        _, (h, _) = self.rnn(X)
        h = h[-1]

        out = self.linear(h)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        speaker, X, y = batch
        y_pred = self(X)

        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        speaker, X, y = batch
        y_pred = self(X)

        loss = F.smooth_l1_loss(y_pred, y)

        self.log("val_loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def predict_step(self, batch):
        speaker, X, y = batch
        y_pred = self(X)

        return y_pred