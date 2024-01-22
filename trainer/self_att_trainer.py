import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
from utils_comm.log_util import logger
from utils_comm.train_util import set_seed, get_device
from projects.data_utils import get_input_df_v1
from NeuroPredPLM.vocabulary import Vocabulary
from NeuroPredPLM.dataset import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.loggers import TensorBoardLogger
from NeuroPredPLM.model_self_att import SimpleBert


class CustomTrainer():
    """
    docstring
    """
    def __init__(self) -> None:
        self.device = get_device()
        set_seed()
        self.df_train, self.df_test = get_input_df_v1()
        self.embedding_type = "vocab"
        self.fixed_vocab_file = 'config/classifier_vocab.json'
        self.category_col_name = 'label'
        self.epochs = 40

    def get_dataset(self):
        vocabulary = Vocabulary.load_vocab_file(self.fixed_vocab_file, max_len=50)
        training_dataset = Dataset(self.df_train, vocabulary, self.category_col_name)
        test_dataset = Dataset(self.df_test, vocabulary, self.category_col_name)
        return training_dataset, test_dataset
    
    def get_dataloader(self):
        training_dataset, test_dataset = self.get_dataset()
        train_loader = data.DataLoader(training_dataset, batch_size=32, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader

    def train(self):
        train_dataloader, val_dataloader = self.get_dataloader()
        pl.seed_everything(42, workers=True)
        train_steps_num_per_epoch = len(train_dataloader)
        total_train_steps = train_steps_num_per_epoch * self.epochs
        logger.info(f'Starts to train, with epochs {self.epochs}')
        logger.info(f"Total train steps per epoch = {train_steps_num_per_epoch}")
        logger.info(f"Total valid steps per epoch = {len(val_dataloader)}")
        logger.info(f"Total train steps = {total_train_steps}")
        warmup_steps = train_steps_num_per_epoch
        logger.info(f'warmup_steps is the train steps in one epoch: {warmup_steps}')
        checkpoint_callback = ModelCheckpoint(
            monitor='test_auprc', mode='max',
            save_top_k=3,
            save_on_train_epoch_end=True,
            filename='{epoch:02d}-{test_auprc:.4f}-{test_auc:.4f}-{test_acc:.4f}-{test_loss:.5f}-{train_loss:.4f}',
        )
        early_stopping = EarlyStopping(
            'test_loss',
            patience=20,
            mode='min',
        )


class LitClassifier(pl.LightningModule):
    """
    docstring
    """
    def __init__(self, lr, max_iters, warmup_steps, seed) -> None:
        super().__init__()
        self.model = SimpleBert()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_iters = max_iters
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self.cal_loss(batch)
        # if both on_epoch=True, on_step=True, the log output have "train_loss_step" and "train_loss_epoch", e.g:
        # Epoch 1:  49%|████▉     | 227/462 [00:41<00:42,  5.48it/s, loss=0.32, v_num=7, train_loss_step=0.255, test_loss=0.406, test_auc=0.710, test_auprc=0.474, test_acc=0.859, train_loss_epoch=0.525]
        # if only on_epoch=True, the log output is:
        # Epoch 18:  74%|███████▍  | 342/462 [00:58<00:20,  5.89it/s, loss=0.0243, v_num=6, test_loss=0.244, test_auc=0.958, test_auprc=0.862, test_acc=0.934, train_loss=0.0527]
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("learning_rate", self.lr_scheduler.get_last_lr()[0], on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        prot_tensor, pep_tensor, y = batch
        y_hat = self.model(prot_tensor, pep_tensor)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        proba = F.softmax(y_hat, dim=1).cpu().to(dtype=torch.float)
        y_proba = proba[:, 1].tolist()
        y = y.cpu().numpy()
        auc = roc_auc_score(y, y_proba)
        auprc = average_precision_score(y, y_proba)
        predicted = torch.argmax(proba, dim=1)
        acc = accuracy_score(y, predicted)
        self.log("test_auc", auc, on_epoch=True, prog_bar=True)
        self.log("test_auprc", auprc, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)


if __name__ == '__main__':
    trainer = CustomTrainer()
