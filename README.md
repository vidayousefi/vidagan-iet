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
 

Generate samples:
```bash
python generate.py --model checkpoints/your_model.pth
```

## Requirements

- Python 3.7+
- PyTorch
- Other dependencies in `requirements.txt`

## License

This project is licensed under the MIT License.

## Acknowledgements

- Based on research in generative modeling and GANs.
- Inspired by leading open-source GAN repositories.
