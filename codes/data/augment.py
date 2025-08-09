from torchvision import transforms
from torchvision.transforms import InterpolationMode

_DEFAULT_MU = 0.5
_DEFAULT_SIGMA = 0.5


class Augmentation:
    train_transform = None
    val_transform = None
    infer_transform = None

    @classmethod
    def _get_train_transform(cls):
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(250, pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
            ]
        )

    @classmethod
    def calc_transform(cls):
        cls.train_transform = cls._get_train_transform()
        cls.val_transform = cls._get_train_transform()
        cls.infer_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
            ]
        )
