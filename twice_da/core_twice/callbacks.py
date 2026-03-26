import torch
from lightning.pytorch.callbacks import Callback

class LossMetricTracker(Callback):
    def __init__(self):
        self.collection = {}

    def on_train_epoch_end(self, trainer, module):
        logs = trainer.logged_metrics
        if(bool(self.collection) == False):
            self.collection['train_accuracy_epoch'] = logs['train_accuracy_epoch'][None].unsqueeze(1)
            self.collection['train_loss_epoch'] = logs['train_loss_epoch'][None].unsqueeze(1)
            self.collection['val_accuracy'] = logs['val_accuracy'][None].unsqueeze(1)
            self.collection['val_loss'] = logs['val_loss'][None].unsqueeze(1)
        else:
            self.collection['train_accuracy_epoch'] = torch.cat((self.collection['train_accuracy_epoch'], logs['train_accuracy_epoch'][None].unsqueeze(1)), dim=1)
            self.collection['train_loss_epoch'] = torch.cat((self.collection['train_loss_epoch'], logs['train_loss_epoch'][None].unsqueeze(1)), dim=1)
            self.collection['val_accuracy'] = torch.cat((self.collection['val_accuracy'], logs['val_accuracy'][None].unsqueeze(1)), dim=1)
            self.collection['val_loss'] = torch.cat((self.collection['val_loss'], logs['val_loss'][None].unsqueeze(1)), dim=1)