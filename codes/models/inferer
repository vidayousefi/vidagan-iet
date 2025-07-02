# -*- coding: utf-8 -*-
import os
from collections import Counter
from os.path import isfile

import imageio.v2 as imageio
import torch
from imageio import imwrite
from PIL import Image
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from tqdm import tqdm

from codes.data.augment import Augmentation
from codes.misc.utils import bits_to_bytearray, bytearray_to_text, text_to_bits
from codes.models.gan import MainGan


class Inferer(object):
    """
    Inferer is a utility class for performing inference with a trained MainGan model, specifically for steganography tasks.
    It provides methods to encode messages into images (stego images), decode messages from images, and generate random stego samples.
        model_file (str): Path to the trained model weights file.
        data_depth (int): Depth of the payload data to be embedded.
        coder (nn.Module): The encoder module for the GAN.
        critic (nn.Module): The critic/discriminator module for the GAN.
    Attributes:
        device (str): The device used for computation ("cuda" or "cpu").
        model (MainGan): The GAN model used for encoding and decoding.
        data_depth (int): The depth of the payload data.
    Methods:
        create_random_stegos(dataloader, dest_path, file_type="jpg"):
            Generates and saves stego images from a dataloader using the model's encoder.
        _forward_coders(cover, quantize=False):
            Passes cover images through the encoder and decoder, returning stego images, payloads, and decoded outputs.
        _forward_encoder(cover, quantize):
            Encodes a cover image with a random payload, optionally quantizing the output.
        encode(cover, output, text):
            Encodes a given text message into a cover image and saves the resulting stego image.
        decode(image):
            Decodes and retrieves the hidden message from a stego image.
        _random_payload(size):
            Generates a random binary payload tensor of the specified size.
        _make_payload_by_text(width, height, depth, text):
            Converts a text message into a bit vector and fills a tensor of the specified shape with the message bits.
    """

    # ============================================== Lifecycle =======================================

    def __init__(self, model_file, data_depth, coder, critic):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MainGan(data_depth, coder, critic, self.device)
        self.model.load_state_dict(torch.load(model_file))
        self.data_depth = data_depth

    # ============================================== Samples =======================================

    def create_random_stegos(self, dataloader, dest_path, file_type="jpg"):
        """
        Generates and saves stego images from a given dataloader using the model's encoder.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of cover images.
            dest_path (str): Directory path where the generated stego images will be saved.
            file_type (str, optional): File extension/type for the saved images (e.g., "jpg", "png"). Defaults to "jpg".

        Notes:
            - The method sets the model to evaluation mode and disables gradient computation.
            - Each batch of cover images is processed by the encoder to produce stego images.
            - Each stego image is clamped to the valid range, converted to a NumPy array, and saved to disk.
            - Images are saved with filenames based on their index in the dataset.
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, cover in enumerate(tqdm(dataloader)):
                cover = cover.to(self.device)
                stego_torch, _ = self._forward_encoder(cover, True)
                batch_size = stego_torch.size(0)
                for i in range(batch_size):
                    im_idx = batch_idx * batch_size + i
                    stego_img = stego_torch[i].clamp(-1.0, 1.0).permute(1, 2, 0)
                    stego_img = (stego_img.detach().cpu().numpy() + 1.0) / 2.0 * 255.0
                    imageio.imwrite(
                        os.path.join(dest_path, f"{im_idx}.{file_type}"),
                        stego_img.astype("uint8"),
                    )

    # ============================================== Forward =======================================

    def _forward_coders(self, cover, quantize=False):
        """
        Encodes the input cover image using the encoder, then decodes the resulting stego image.
        Args:
            cover (Tensor): The input cover image tensor to be encoded.
            quantize (bool, optional): If True, applies quantization during encoding. Defaults to False.
        Returns:
            tuple: A tuple containing:
                - stego (Tensor): The encoded stego image tensor.
                - payload (Tensor): The payload tensor produced by the encoder.
                - decoded (Tensor): The decoded output tensor from the decoder.
        """
        stego, payload = self._forward_encoder(cover, quantize)

        decoded = self.model.decoder(stego, None)

        return stego, payload, decoded

    def _forward_encoder(self, cover, quantize):
        """
        Encodes a random payload into the given cover image using the model's encoder.

        Args:
            cover (torch.Tensor): The input cover image tensor.
            quantize (bool): If True, applies quantization to the encoded (stego) image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - stego: The encoded (stego) image tensor, possibly quantized.
                - payload: The randomly generated payload tensor that was embedded.
        """
        payload = self._random_payload(cover.size())
        stego = self.model.encoder(cover, payload)
        if quantize:
            stego = (255.0 * (stego + 1.0) / 2.0).long()
            stego = 2.0 * stego.float() / 255.0 - 1.0
        return stego, payload

    # ============================================== Prediction =======================================

    def encode(self, cover, output, text):
        """
        Encodes a text message into a cover image using the model's encoder and saves the resulting stego image.
        Args:
            cover (str): Path to the cover image file.
            output (str): Path where the encoded (stego) image will be saved.
            text (str): The text message to encode into the image.
        Steps:
            1. Loads and preprocesses the cover image.
            2. Generates a payload tensor from the input text.
            3. Passes the cover image and payload through the encoder model to produce the stego image.
            4. Post-processes and saves the stego image to the specified output path.
            5. Prints a completion message.
        Note:
            The method assumes that the model, device, data_depth, and augmentation transforms are properly initialized.
        """
        cover = Augmentation.val_transform(Image.open(cover).convert("RGB"))
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0).to(self.device)

        cover_size = cover.size()
        payload = self._make_payload_by_text(
            cover_size[3], cover_size[2], self.data_depth, text
        ).to(self.device)

        stego = self.model.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        stego = (stego.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, stego.astype("uint8"))

        print("Encoding completed.")

    def decode(self, image):
        """
        Decodes a hidden message from an input image using the model's decoder.
        Args:
            image (str or Path): Path to the image file from which to decode the message.
        Returns:
            str: The most common decoded message extracted from the image.
        Raises:
            ValueError: If no valid message could be found in the image.
        """
        # extract a bit vector
        image = Augmentation.val_transform(Image.open(image).convert("RGB"))
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0).to(self.device)
        image = self.model.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b"\x00\x00\x00\x00"):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError("Failed to find message.")

        candidate, count = candidates.most_common(1)[0]
        return candidate

    # ============================================== Payload =======================================

    def _random_payload(self, size):
        N, _, H, W = size
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    @staticmethod
    def _make_payload_by_text(width, height, depth, text):
        """
        Encodes a given text string into a bit vector and fills a tensor of shape (1, depth, height, width) with repeated copies of this bit vector.
        Args:
            width (int): The width of the output tensor.
            height (int): The height of the output tensor.
            depth (int): The depth (number of channels) of the output tensor.
            text (str): The input text to encode into the payload.
        Returns:
            torch.FloatTensor: A tensor of shape (1, depth, height, width) containing the encoded text as a repeated bit vector.
        Note:
            The text is first converted to a bit vector using `text_to_bits`, then padded with 32 zeros. The resulting bit vector is repeated to fill the entire tensor, and truncated if necessary.
        """
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[: width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)
