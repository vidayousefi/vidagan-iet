# -*- coding: utf-8 -*-
import json
import math
import os
import pickle
import time
from collections import Counter
from os.path import isfile

import imageio.v2 as imageio
import torch
from PIL import Image
from imageio import imwrite
from prettytable import PrettyTable
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.augment import Augmentation
from misc.utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits, linear_fit
from models.gan import MainGan
from optimization.scheduler import CustomScheduler, SchedulerStage

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


class Trainer(object):
    writer = None
    critic_optimizer = critic_scheduler = decoder_optimizer = decoder_scheduler = None

    # ============================================== Lifecycle =======================================

    def __init__(self, data_depth, coder, critic, log_dir, writer_dir, net_dir, sample_dir):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MainGan(data_depth, coder, critic, self.device)
        self.data_depth = data_depth
        self.metric_history = []
        self.encoder_mse_history = []
        self.epoch = 0
        self.after_deserialize(log_dir, writer_dir, net_dir, sample_dir)

    # noinspection PyAttributeOutsideInit
    def after_deserialize(self, log_dir, writer_dir, net_dir, sample_dir):
        self.log_dir = log_dir
        self.net_dir = net_dir
        self.sample_dir = sample_dir
        self.epoch += 1
        Trainer.writer = SummaryWriter(writer_dir)

    def _save_self(self, path):
        with open(path, 'wb') as fs:
            pickle.dump(self, fs)

    @classmethod
    def load(cls, trainer_file, model_file):
        with open(trainer_file, 'rb') as fs:
            obj = pickle.load(fs)
        obj.model.load_state_dict(torch.load(model_file))
        return obj

    # ============================================== Train =======================================

    def fit(self, train_loader, val_loader, epochs=5):

        Trainer.critic_optimizer, Trainer.critic_scheduler, Trainer.decoder_optimizer, Trainer.decoder_scheduler \
            = self._get_optimizers(2e-4, 1.5e-4, self.epoch, epochs, len(train_loader))

        self.encoder_mse_history = []
        sample_cover_batch = next(iter(val_loader))
        total_coder_time = 0
        start_epoch = self.epoch
        batch_size = sample_cover_batch.size()[0]

        self.count_parameters(self.model.decoder)

        for ep in range(self.epoch, epochs + 1):
            self.epoch = ep
            print(f'Epoch {ep}/{epochs}')

            metrics = {field: list() for field in METRIC_FIELDS}

            self._fit_critic(train_loader, metrics, ep)

            start_time = time.perf_counter()
            self._fit_coders(train_loader, metrics, ep, epochs)
            end_time = time.perf_counter()
            total_coder_time += end_time - start_time
            print(f'FPS: {len(train_loader) * (ep - start_epoch + 1) * batch_size / total_coder_time:.2f}')

            self._validate(val_loader, metrics)

            aggregate_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            aggregate_metrics['epoch'] = ep
            self.metric_history.append(aggregate_metrics)

            # log tensorboard
            self.log_tensorboard(ep, 1, metrics, ['val.encoder_mse', 'val.decoder_acc', 'val.ssim', 'val.psnr',
                                                  'val.bpp'])

            # log metrics
            metrics_path = os.path.join(self.log_dir, 'metrics.log')
            with open(metrics_path, 'w') as metrics_file:
                json.dump(self.metric_history, metrics_file, indent=4)

            # serialize self
            self._save_self(os.path.join(self.net_dir, f'trainer-{ep}.bin'))

            # save model
            torch.save(self.model.state_dict(), os.path.join(self.net_dir, f'model-{ep}.pth'))

            # generate sample images in samples folder
            if ep == epochs:
                self._create_sample_results(self.sample_dir, val_loader)

        Trainer.writer.close()

    def _fit_critic(self, dataloader, metrics, epoch):
        self.model.train()

        for idx, cover in enumerate(tqdm(dataloader)):
            cover = cover.to(self.device)
            payload = self._random_payload(cover.size())
            stego = self.model.encoder(cover, payload)
            real_score = self._forward_critic(cover)
            fake_score = self._forward_critic(stego)

            Trainer.critic_optimizer.zero_grad()
            (real_score - fake_score).backward(retain_graph=False)
            Trainer.critic_optimizer.step()
            Trainer.critic_scheduler.step()

            # print(Trainer.critic_scheduler.get_last_lr())

            for p in self.model.critic_params():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(real_score.item())
            metrics['train.generated_score'].append(fake_score.item())

            self.log_tensorboard(len(dataloader) * (epoch - 1) + idx + 1, 5, metrics, ['train.cover_score',
                                                                                       'train.generated_score'])

    def _fit_coders(self, dataloader, metrics, epoch, epochs):
        self.model.train()

        for idx, cover in enumerate(tqdm(dataloader)):
            cover = cover.to(self.device)
            stego, payload, decoded = self._forward_coders(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coders_loss(cover, stego, payload, decoded)
            realness_loss = self._forward_critic(stego)
            self._calc_encoder_ratio(encoder_mse, 0.0006, epoch > epochs - 4)

            Trainer.decoder_optimizer.zero_grad()
            (encoder_mse + (decoder_loss + realness_loss) * self.ratio).backward()
            # (encoder_mse + (decoder_loss + realness_loss) * 0.01).backward()
            Trainer.decoder_optimizer.step()
            Trainer.decoder_scheduler.step()

            Trainer.writer.add_scalar('lr', Trainer.decoder_scheduler.get_last_lr()[0],
                len(dataloader) * self.epoch + idx)

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

            self.log_tensorboard(len(dataloader) * (epoch - 1) + idx + 1, 5, metrics, ['train.encoder_mse',
                                                                                       'train.decoder_loss',
                                                                                       'train.decoder_acc'])

    # ============================================== Validation =======================================

    def _validate(self, validate, metrics):
        self.model.eval()

        with torch.no_grad():
            for cover in tqdm(validate):
                cover = cover.to(self.device)
                stego, payload, decoded = self._forward_coders(cover, quantize=True)
                encoder_mse, decoder_loss, decoder_acc = self._coders_loss(cover, stego, payload, decoded)
                generated_score = self._forward_critic(stego)
                cover_score = self._forward_critic(cover)

                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.generated_score'].append(generated_score.item())
                metrics['val.ssim'].append(ssim(cover, stego).item())
                metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
                metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    # ============================================== Forward =======================================

    def _forward_coders(self, cover, quantize=False):
        payload = self._random_payload(cover.size())
        stego = self.model.encoder(cover, payload)
        if quantize:
            stego = (255.0 * (stego + 1.0) / 2.0).long()
            stego = 2.0 * stego.float() / 255.0 - 1.0

        decoded = self.model.decoder(stego, None)

        return stego, payload, decoded

    def _forward_critic(self, image):
        """Evaluate the image using the critic"""
        return torch.mean(self.model.critic(image))

    # ============================================== Loss =======================================

    @staticmethod
    def _coders_loss(cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_sigmoid = sigmoid(decoded)
        soft_label_loss = 0.15 * ((decoder_sigmoid * 2 - 1) ** 4).mean()
        decoder_loss = decoder_loss + soft_label_loss
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _calc_encoder_ratio(self, encoder_mse, tgt, smoother):
        target_mse = math.log(tgt, 10)
        mse_hist = self.encoder_mse_history
        mse_hist.append(math.log(encoder_mse.item(), 10))
        points_count = 80
        if len(mse_hist) < points_count:
            self.ratio = .01
        else:
            mse_hist = mse_hist[-points_count:]

            xs = 0.2
            mp, bp = linear_fit([i * xs for i in range(points_count)], mse_hist)
            mp = min(max(mp, -1), 1)
            # mt, bt = linear_fit(
            #     [(points_count - 1) / 2 * xs, ((points_count - 1) / 2 + 60 * (6 if smoother else 2)) * xs],
            #     [sum(mse_hist) / points_count, target_mse])
            mt, bt = linear_fit(
                [(points_count - 1) / 2 * xs, ((points_count - 1) / 2 + 60 * 2) * xs],
                [sum(mse_hist) / points_count, target_mse])
            mt = min(max(mt, -1), 1)
            self.ratio *= 1 + (mt - mp)
            self.ratio = min(max(self.ratio, 0.005), 2)

    # ============================================== Prediction =======================================

    def encode(self, cover, output, text):
        cover = Augmentation.val_transform(Image.open(cover).convert("RGB"))
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0).to(self.device)

        cover_size = cover.size()
        payload = self._make_payload_by_text(cover_size[3], cover_size[2], self.data_depth, text).to(self.device)

        stego = self.model.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        stego = (stego.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, stego.astype('uint8'))

        print('Encoding completed.')

    def decode(self, image):
        # extract a bit vector
        image = Augmentation.val_transform(Image.open(image).convert("RGB"))
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0).to(self.device)
        image = self.model.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]
        return candidate

    # ============================================== Samples =======================================

    # def _create_sample_results(self, samples_path, cover_batch, epoch):
    def _create_sample_results(self, samples_path, dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, cover in enumerate(tqdm(dataloader)):
                cover = cover.to(self.device)
                stego_torch, payload, decoded = self._forward_coders(cover)
                batch_size = stego_torch.size(0)
                for i in range(batch_size):
                    im_idx = batch_idx * batch_size + i
                    cover_path = os.path.join(samples_path, f'{im_idx}.cover.png')
                    if not isfile(cover_path):
                        cover_img = (cover[i].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0 * 255.
                        imageio.imwrite(cover_path, cover_img.astype('uint8'))

                    stego_img = stego_torch[i].clamp(-1.0, 1.0).permute(1, 2, 0)
                    stego_img = (stego_img.detach().cpu().numpy() + 1.0) / 2. * 255.
                    imageio.imwrite(os.path.join(samples_path, f'stego-{im_idx}.png'), stego_img.astype('uint8'))

    # ============================================== Optimiser =======================================

    def _get_optimizers(self, lr, weight_decay, start_epoch, total_epochs, iters_per_epoch):
        critic_optimizer, critic_scheduler = self._create_optimizer(self.model.critic_params(), lr, weight_decay,
            start_epoch, total_epochs, iters_per_epoch)

        decoder_optimizer, decoder_scheduler = self._create_optimizer(self.model.coder_params(), lr, weight_decay,
            start_epoch, total_epochs, iters_per_epoch)

        return critic_optimizer, critic_scheduler, decoder_optimizer, decoder_scheduler

    @staticmethod
    def _create_optimizer(parameters, lr, weight_decay, start_epoch, total_epochs, iters_per_epoch):
        # if total_iterations > 0 and warmup_iterations > 0:
        #     optimizer = SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        #     if warmup_iterations > 0:
        #         scheduler = LambdaLR(
        #             optimizer, lambda i: min(i / warmup_iterations, 1) * cos(
        #                 max(i, warmup_iterations) / total_iterations * pi / 2))
        #     else:
        #         scheduler = CosineAnnealingLR(optimizer, total_iterations)
        # else:
        #     optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
        #     if warmup_iterations > 0:
        #         scheduler = LambdaLR(
        #             optimizer, lambda i: min(i / warmup_iterations, 1) * cos(
        #                 max(i, warmup_iterations) / total_iterations * pi / 2))
        #     else:
        #         scheduler = CosineAnnealingLR(optimizer, total_iterations)
        #     # scheduler = ConstantLR(optimizer, factor=1, total_iters=total_iterations)
        optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
        scheduler = CustomScheduler(optimizer, [
            SchedulerStage('linear', (0.1, 1), 1),
            SchedulerStage('linear', (1, 0.75), 2, True),
            # SchedulerStage('constant', 1, 2, True),
            # SchedulerStage('constant', 1, 2, True),
            # SchedulerStage('linear', (1, 0.3), 10),
            # SchedulerStage('linear', (0.3, 1), 12),
            # SchedulerStage('constant', 1, 14, True),
            # SchedulerStage('linear', (1, 0.3), 16),
            # SchedulerStage('linear', (0.3, 1), 18),
            # SchedulerStage('cosine', 1, 2, True),
        ], total_epochs, iters_per_epoch, start_epoch)

        return optimizer, scheduler

    # ============================================== Logging =======================================

    @staticmethod
    def log_tensorboard(idx, interval, metrics_list, scores):
        if idx % interval == 0:
            for score in scores:
                lst = metrics_list[score][-interval:]
                Trainer.writer.add_scalar(score, sum(lst) / len(lst), idx)

    # ============================================== Payload =======================================

    def _random_payload(self, size):
        N, _, H, W = size
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    @staticmethod
    def _make_payload_by_text(width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        """
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)

    # ============================================== Manipulation =======================================

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
