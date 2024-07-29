import abc
import gc
from typing import Callable, Dict, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from modules import *

class BaseClassifier(pl.LightningModule, abc.ABC):

    classifier: nn.Module  # classifier mapping from ncell_dim to type_dim - outputs logits

    def __init__(
        self,
        # fixed params
        ncell_dim: int,
        label_str: str,
        num_classes: int,
        class_weights: np.ndarray,
        # params from datamodule
        train_set_size: int,
        val_set_size: int,
        batch_size: int,
        # model specific params
        learning_rate: float = 0.005,
        weight_decay: float = 0.05,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 10
    ):
        super(BaseClassifier, self).__init__()

        self.ncell_dim = ncell_dim
        self.label_str: label_str
        self.num_classes = num_classes
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size
        self.gc_freq = gc_frequency
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        metrics = MetricCollection({
            'f1_micro': MulticlassF1Score(num_classes=self.num_classes, average='micro'),
            'f1_macro': MulticlassF1Score(num_classes=self.num_classes, average='macro'),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.register_buffer('class_weights', torch.tensor(class_weights.astype('f4')))

    @abc.abstractmethod
    def _step(self, batch, training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        with torch.no_grad():
            batch = batch[0]
            batch['cell_type'] = torch.squeeze(batch['cell_type'])

        return batch

    def forward(self, x: torch.Tensor):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=True)
        self.log('train_loss', loss)
        f1_macro = self.train_metrics['f1_macro'](preds, batch['cell_type'])
        f1_micro = self.train_metrics['f1_micro'](preds, batch['cell_type'])
        self.log('train_f1_macro_step', f1_macro)
        self.log('train_f1_micro_step', f1_micro)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=False)
        self.log('val_loss', loss)
        self.val_metrics['f1_macro'].update(preds, batch['cell_type'])
        self.val_metrics['f1_micro'].update(preds, batch['cell_type'])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=False)
        self.log('test_loss', loss)
        self.test_metrics['f1_macro'].update(preds, batch['cell_type'])
        self.test_metrics['f1_micro'].update(preds, batch['cell_type'])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def on_train_epoch_end(self) -> None:
        self.log('train_f1_macro_epoch', self.train_metrics['f1_macro'].compute())
        self.train_metrics['f1_macro'].reset()
        self.log('train_f1_micro_epoch', self.train_metrics['f1_micro'].compute())
        self.train_metrics['f1_micro'].reset()
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        f1_macro = self.val_metrics['f1_macro'].compute()
        self.log('val_f1_macro', f1_macro)
        self.log('hp_metric', f1_macro)
        self.val_metrics['f1_macro'].reset()
        self.log('val_f1_micro', self.val_metrics['f1_micro'].compute())
        self.val_metrics['f1_micro'].reset()
        gc.collect()

    def on_test_epoch_end(self) -> None:
        self.log('test_f1_macro', self.test_metrics['f1_macro'].compute())
        self.test_metrics['f1_macro'].reset()
        self.log('test_f1_micro', self.test_metrics['f1_micro'].compute())
        self.test_metrics['f1_micro'].reset()
        gc.collect()

    def configure_optimizers(self):
        optimizer_config = {'optimizer': self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)}
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            interval = lr_scheduler_kwargs.pop('interval', 'epoch')
            monitor = lr_scheduler_kwargs.pop('monitor', 'val_loss_epoch')
            frequency = lr_scheduler_kwargs.pop('frequency', 1)
            scheduler = self.lr_scheduler(optimizer_config['optimizer'], **lr_scheduler_kwargs)
            optimizer_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': interval,
                'monitor': monitor,
                'frequency': frequency
            }

        return optimizer_config

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
    
class SetTransformerClassifier(BaseClassifier):
    def __init__(self, 
        # SetTransformer args
        dim_input, num_outputs, dim_output,
        num_inds=32, dim_hidden=128, num_heads=4, ln=False, **kwargs):
        super(SetTransformerClassifier, self).__init__(**kwargs)
        self.classifier = SetTransformer(dim_input, num_outputs, dim_output,
                num_inds=num_inds, dim_hidden=dim_hidden, num_heads=num_heads, ln=ln)

    def _step(self, batch, training=True):
        x = batch['x']
        y = batch[self.label_str]
        preds = self(x)
        loss = F.cross_entropy(preds, y, weight=self.class_weights)

        return preds, loss