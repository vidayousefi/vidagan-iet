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
from codes.models.inferer import Inferer
from codes.models.trainer import Trainer


def prepare_data(dir):
    dataset = Div2kDataset(dir, Augmentation.infer_transform)
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    file_names = dataset.img_paths
    return loader, file_names


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
    os.makedirs(os.path.join(dest_dir, "cover"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "stego"), exist_ok=True)

    Augmentation.calc_transform()
    val_loader, file_names = prepare_data(args.source_path)

    inferer = Inferer(
        model_file=args.model_path,
        data_depth=args.data_depth,
        coder=CSPDenseCoder,
        critic=BasicCritic,
    )

    inferer.create_random_stegos(val_loader, dest_dir, file_names)


if __name__ == "__main__":
    main()
