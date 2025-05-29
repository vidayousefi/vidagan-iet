# v5
import argparse
import os

import torch
from torch.utils.data import DataLoader

from codes.data.augment import Augmentation
from codes.misc.utils import get_unique_file
from codes.models.coders.dense_coder import CSPDenseCoder
from codes.models.trainer import Trainer
from codes.models.critics import BasicCritic
from codes.data.loader import Div2kDataset


def prepare_data(data_dir):
    train_data = Div2kDataset(os.path.join(data_dir, "train", "_"), Augmentation.train_transform)
    train = DataLoader(train_data, batch_size=4, num_workers=4, shuffle=True)
    validation_data = Div2kDataset(os.path.join(data_dir, "val", "_"), Augmentation.val_transform)
    validation = DataLoader(validation_data, batch_size=4, num_workers=4, shuffle=False)
    return train, validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--data_depth', default=1, type=int)
    parser.add_argument('--dataset', default="div2k", type=str)
    parser.add_argument('--augmentation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--resume-epoch', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    torch.manual_seed(42)

    args = parse_args()

    results_dir = 'results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    data_dir = os.path.join('dataset', args.dataset)
    log_dir = os.path.join(results_dir, get_unique_file('results', 100))
    writer_dir = os.path.join(log_dir, 'tensorboard')
    net_dir = os.path.join(log_dir, 'network')
    sample_dir = os.path.join(log_dir, 'samples')

    os.makedirs(log_dir)
    os.mkdir(net_dir)
    os.makedirs(sample_dir)

    Augmentation.calc_transform(args.augmentation)
    train, validation = prepare_data(data_dir)

    resume = args.resume
    if resume:
        epoch = args.resume_epoch
        trainer = Trainer.load(os.path.join(resume, 'network', f'trainer-{epoch}.bin'),
                               os.path.join(resume, 'network', f'model-{epoch}.pth'))
        trainer.after_deserialize(log_dir, writer_dir, net_dir, sample_dir)
    else:
        trainer = Trainer(data_depth=args.data_depth, coder=CSPDenseCoder, critic=BasicCritic,
                          log_dir=log_dir, writer_dir=writer_dir, net_dir=net_dir, sample_dir=sample_dir)

    trainer.fit(train, validation, epochs=args.epochs)

    torch.save(trainer.model.state_dict(), os.path.join(results_dir, 'latest.pth'))


if __name__ == '__main__':
    main()
