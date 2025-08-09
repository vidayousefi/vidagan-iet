# vidaGAN

vidaGAN is a deep learning project focused on Generative Adversarial Networks (GANs) for image and video synthesis. This repository contains code, models, and documentation to help you train and evaluate GANs for various generative tasks.

## Features

- Modular GAN architecture for images and videos
- Training and evaluation scripts
- Preprocessing utilities
- Sample datasets and results

## Installation

```bash
git clone https://github.com/mrtkhosravi/VidaGAN.git
cd vidaGAN
pip install -r requirements.txt
```

## Usage

Train a GAN model:
```bash
python train.py --train_dataset=PATH_TO_TRAIN_IMAGES --val_dataset=PATH_TO_VAL_IMAGES
```

Additional Parameters:

 --epochs,          default=16

 --data_depth,      default=6
 
 --batch_size,      default=4
 

Generate random stego images:
```bash
python inference.py --source_path=COVER_IMAGE_DIR --dest_path=GENERATED_STEGO_DIR --model_path=PATH_TO_TRAIED_MODEL --data_depth=DATA_DEPTH_OF_TRAINED_MODEL
```

## License

This project is licensed under the MIT License.
