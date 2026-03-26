import torch
import lightning.pytorch as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn as nn

class ModelCompilation(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 metrics: dict,
                 loss_function,
                 optimizer: torch.optim,
                 learning_rate: float,
                 accelerator: str,
                 data_module: pl.LightningDataModule):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.accelerator = accelerator
        self.data_module = data_module
        self.metrics = nn.ModuleDict({
            stage: metrics.clone(prefix=stage) for stage in ["train_", "val_", "test_"]
        })

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        train_optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, weight_decay=0.05, betas=(0.9, 0.98), eps=1.0e-9)
        scheduler = CosineAnnealingWarmRestarts(train_optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        return {
            'optimizer': train_optimizer,
            'lr_scheduler': {'scheduler': scheduler,
                             'interval': 'epoch',  # Update the scheduler every epoch
                             'frequency': 1  # Apply the scheduler every epoch
                             }
        }

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, 'test')
        return loss

    def common_step(self, batch, batch_idx, stage):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)

        on_step = False if (stage == 'test') or (stage == 'val') else True
        stage_prefix = stage + "_"
        current_metrics = self.metrics[stage_prefix]
        current_metrics(y_pred, y)

        self.log_dict(current_metrics, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        self.log(stage + '_' + 'loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)
        if stage == 'train':
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss