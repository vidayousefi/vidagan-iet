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


def prepare_data(val_data_dir):
    validation_data = Div2kDataset(val_data_dir, Augmentation.val_transform)
    validation = DataLoader(validation_data, batch_size=4, num_workers=4, shuffle=False)
    return validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_depth", default=6, type=int)
    parser.add_argument("--source_path", default="", type=str)
    parser.add_argument("--dest_path", default="", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()
    return args


def main():
    torch.manual_seed(42)

    args = parse_args()

    dest_dir = args.dest_path
    os.makedirs(dest_dir, exist_ok=True)

    Augmentation.calc_transform()
    val_loader = prepare_data(args.source_path)

    trainer = Trainer(
        data_depth=args.data_depth,
        coder=CSPDenseCoder,
        critic=BasicCritic,
        log_dir=None,
        writer_dir=writer_dir,
        net_dir=net_dir,
        sample_dir=sample_dir,
        dev_mode=args.dev_mode,
    )

    trainer.fit(train, val_loader, epochs=args.epochs)

    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "latest.pth"))


if __name__ == "__main__":
    main()
