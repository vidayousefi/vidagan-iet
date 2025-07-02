# v7
import argparse
import os

import torch
from torch.utils.data import DataLoader

from codes.data.augment import Augmentation
from codes.data.loader import Div2kDataset
from codes.misc.utils import get_unique_file
from codes.models.coders.dense_coder import CSPDenseCoder
from codes.models.critics import BasicCritic
from codes.models.trainer import Trainer


def prepare_data(train_data_dir, val_data_dir):
    train_data = Div2kDataset(train_data_dir, Augmentation.train_transform)
    train = DataLoader(train_data, batch_size=4, num_workers=4, shuffle=True)

    validation_data = Div2kDataset(val_data_dir, Augmentation.val_transform)
    validation = DataLoader(validation_data, batch_size=4, num_workers=4, shuffle=False)
    return train, validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=16, type=int)
    parser.add_argument("--data_depth", default=6, type=int)
    parser.add_argument("--source_path", default="", type=str)
    parser.add_argument("--dest_path", default="", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument(
        "--dev_mode", default=False, action=argparse.BooleanOptionalAction
    )
    # parser.add_argument('--resume', type=str)
    # parser.add_argument('--resume-epoch', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    torch.manual_seed(42)

    args = parse_args()

    results_dir = "results"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    log_dir = os.path.join(results_dir, get_unique_file("results", 100))
    writer_dir = os.path.join(log_dir, "tensorboard")
    net_dir = os.path.join(log_dir, "network")
    sample_dir = os.path.join(log_dir, "samples")

    os.makedirs(log_dir)
    os.mkdir(net_dir)
    os.makedirs(sample_dir)

    Augmentation.calc_transform()
    train, validation = prepare_data(args.train_dataset, args.val_dataset)

    trainer = Trainer(
        data_depth=args.data_depth,
        coder=CSPDenseCoder,
        critic=BasicCritic,
        log_dir=log_dir,
        writer_dir=writer_dir,
        net_dir=net_dir,
        sample_dir=sample_dir,
        dev_mode=args.dev_mode,
    )

    trainer.fit(train, validation, epochs=args.epochs)

    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "latest.pth"))


if __name__ == "__main__":
    main()
