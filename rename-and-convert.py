import os

import imageio
import numpy as np
from PIL import Image


def convert_png_to_jpg(source_dir, dest_cover_dir, dest_stego_dir):
    # Walk through all folders and subfolders
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:

            fname = filename.lower()
            is_cover = "cover" in fname
            fname = fname.replace(".cover", "").replace("stego-", "")

            jpg_name = os.path.splitext(fname)[0] + '.jpg'
            # jpg_name = os.path.splitext(fname)[0] + '.png'
            jpg_path = os.path.join(dest_cover_dir if is_cover else dest_stego_dir, jpg_name)

            source_path = os.path.join(dirpath, filename)
            try:
                # Open and convert the image
                with Image.open(source_path) as img:
                    # Convert to RGB if image is RGBA
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    # Crop and Save as JPG
                    stego_img = np.array(img)
                    imageio.imwrite(jpg_path, stego_img[:256, :256].astype("uint8"))
                    print(f"Converted: {source_path} -> {jpg_path}")

            except Exception as e:
                print(f"Error converting {source_path}: {str(e)}")


if __name__ == "__main__":
    # Specify the root directory to start conversion
    source_dir = "D:/Github/vidagan2/results/037-8 bit/samples"
    dest_dir = "D:/Datasets/Vidaformer/8-bit"

    dest_cover_dir = os.path.join(dest_dir, 'cover')
    os.makedirs(dest_cover_dir, exist_ok=True)

    dest_stego_dir = os.path.join(dest_dir, 'stego')
    os.makedirs(dest_stego_dir, exist_ok=True)

    convert_png_to_jpg(source_dir, dest_cover_dir, dest_stego_dir)
