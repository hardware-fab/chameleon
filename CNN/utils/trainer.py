"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning.callbacks.early_stopping as early_stopping


def build_trainer(exp_dir, exp_config, neptune_logger, gpu):

    callbacks = []

    # Build Model Checkpoint
    # ----------------------
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, 'checkpoints'),
        filename='epoch={epoch}-step={step}-loss={valid/loss:.4f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=exp_config['checkpoint']['monitor'],
        mode=exp_config['checkpoint']['mode'])
    callbacks.append(model_ckpt)
    # ----------------------

    # Build Early Stopping
    # --------------------
    if 'early_stop' in exp_config:
        early_stop_name = exp_config['early_stop']['name']
        early_stop_config = exp_config['early_stop']['config']
        early_stop_class = getattr(early_stopping, early_stop_name)
        early_stop = early_stop_class(**early_stop_config)
        callbacks.append(early_stop)
    # --------------------

    # Build LR Monitor
    # --------------------
    lr_monitor = LearningRateMonitor(logging_interval=None)
    callbacks.append(lr_monitor)
    # --------------------
    

    # Build Trainer
    # -------------
    trainer_config = exp_config['trainer']
    trainer = pl.Trainer(
        accelerator="gpu", devices=[gpu],
        max_epochs=trainer_config['max_epochs'],
        check_val_every_n_epoch=trainer_config['check_val_every_n_epoch'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        limit_train_batches=trainer_config['limit_train_batches'],
        limit_val_batches=trainer_config['limit_val_batches'],
        callbacks=callbacks,
        logger=neptune_logger
    )
    # -------------
    return trainer
