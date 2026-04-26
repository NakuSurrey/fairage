"""
image preprocessing pipelines.

two pipelines:
    train     — light augmentation, helps generalisation
    inference — deterministic, no randomness, used at eval and serving time

both pipelines end with imagenet normalisation because the ResNet-50 backbone
is pretrained on imagenet and expects those statistics.
"""

from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform() -> transforms.Compose:
    """
    training pipeline.

    augmentations chosen to match real-world deployment conditions:
    - horizontal flip — face can appear mirrored on different cameras
    - small rotation — natural head tilt
    - colour jitter — different lighting and white balance
    no vertical flip — upside-down faces are not a real deployment case.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),  # converts PIL [0-255] -> float tensor [0.0-1.0]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform() -> transforms.Compose:
    """
    inference pipeline — deterministic, no randomness.
    used for validation, test set evaluation, bias audit, and live API serving.
    output must be byte-identical for the same input every run.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
