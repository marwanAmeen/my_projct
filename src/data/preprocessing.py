"""Data preprocessing utilities"""
import torch
import torchvision.transforms as transforms
from typing import Optional, Tuple
import numpy as np
from PIL import Image


def get_train_transforms(image_size: int = 224, 
                         mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                         std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
    """
    Get training image transforms with augmentation
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(image_size: int = 224,
                       mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                       std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
    """
    Get validation/test image transforms without augmentation
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_image(image: Image.Image, 
                     image_size: int = 224,
                     normalize: bool = True) -> torch.Tensor:
    """
    Preprocess a single image
    
    Args:
        image: PIL Image
        image_size: Target size
        normalize: Whether to normalize
    
    Returns:
        Preprocessed image tensor
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image)


def preprocess_text(text: str, lower: bool = True, remove_punct: bool = False) -> str:
    """
    Preprocess text (questions/answers)
    
    Args:
        text: Input text
        lower: Convert to lowercase
        remove_punct: Remove punctuation
    
    Returns:
        Preprocessed text
    """
    if lower:
        text = text.lower()
    
    if remove_punct:
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def denormalize_image(tensor: torch.Tensor,
                      mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                      std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean used
        std: Normalization std used
    
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clip to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    return (image * 255).astype(np.uint8)


def split_train_val(csv_file: str, val_split: float = 0.15, random_state: int = 42):
    """
    Split training data into train and validation sets
    
    Args:
        csv_file: Path to training CSV
        val_split: Fraction of data for validation
        random_state: Random seed
    
    Returns:
        train_df, val_df: DataFrames for train and validation
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_file)
    
    # Split by images to avoid data leakage
    unique_images = df['image'].unique()
    train_images, val_images = train_test_split(
        unique_images, 
        test_size=val_split, 
        random_state=random_state
    )
    
    train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['image'].isin(val_images)].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} samples from {len(train_images)} images")
    print(f"Val: {len(val_df)} samples from {len(val_images)} images")
    
    return train_df, val_df
