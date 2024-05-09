"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python test.py --ckpt_file wandb/aue8088-pa1/ygeiua2t/checkpoints/epoch\=19-step\=62500.ckpt
"""
# Python packages
import argparse
import torch
# PyTorch & Pytorch Lightning
from lightning import Trainer

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg
from src.flopcounter import FlopCounterMode

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt_file',
        type = str,
        help = 'Model checkpoint file name')
    args = args.parse_args()

    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = 1,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        benchmark = True,
        inference_mode = True,
        logger = False,
    )

    trainer.validate(model, ckpt_path = args.ckpt_file, datamodule = datamodule)

    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)
