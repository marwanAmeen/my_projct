"""
Vision encoder module for extracting features from medical images
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class CNNVisionEncoder(nn.Module):
    """
    CNN-based vision encoder for extracting features from pathology images.
    Uses pre-trained ResNet as backbone.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            feature_dim: Output feature dimension
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: Freeze backbone weights (only train final layer)
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load pre-trained ResNet
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer to desired feature dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.backbone_dim = backbone_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            images: Tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            features: Tensor of shape [batch_size, feature_dim]
        """
        # Extract features from backbone
        features = self.backbone(images)  # [batch, backbone_dim, 1, 1]
        
        # Project to desired dimension
        features = self.projection(features)  # [batch, feature_dim]
        
        return features
    
    def get_attention_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get spatial features for attention mechanisms
        
        Args:
            images: Tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            features: Tensor of shape [batch_size, backbone_dim, 7, 7]
        """
        # Get features before final pooling
        features = self.backbone[:-1](images)  # Remove AdaptiveAvgPool
        return features


class AttentionVisionEncoder(nn.Module):
    """
    Vision encoder with spatial attention for focusing on relevant image regions
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 512,
        num_attention_heads: int = 8,
        pretrained: bool = True
    ):
        """
        Args:
            backbone: ResNet variant
            feature_dim: Output feature dimension
            num_attention_heads: Number of attention heads
            pretrained: Use pre-trained weights
        """
        super().__init__()
        
        # Base vision encoder
        self.encoder = CNNVisionEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            freeze_backbone=False
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder.backbone_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Spatial projection
        self.spatial_projection = nn.Linear(self.encoder.backbone_dim, feature_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract attention-weighted features
        
        Args:
            images: Tensor of shape [batch_size, 3, 224, 224]
            text_features: Optional text features for cross-modal attention
                          Shape: [batch_size, text_dim]
        
        Returns:
            features: Tensor of shape [batch_size, feature_dim]
        """
        # Get spatial features [batch, channels, H, W]
        spatial_features = self.encoder.get_attention_features(images)
        
        batch_size, channels, height, width = spatial_features.shape
        
        # Reshape to [batch, H*W, channels] for attention
        spatial_features = spatial_features.view(batch_size, channels, -1)
        spatial_features = spatial_features.permute(0, 2, 1)  # [batch, H*W, channels]
        
        if text_features is not None:
            # Cross-modal attention: text queries image
            query = text_features.unsqueeze(1)  # [batch, 1, text_dim]
            
            # Project text to same dimension as spatial features
            if text_features.size(-1) != channels:
                query = nn.Linear(text_features.size(-1), channels).to(text_features.device)(query)
            
            attended_features, _ = self.attention(
                query,
                spatial_features,
                spatial_features
            )
            attended_features = attended_features.squeeze(1)  # [batch, channels]
        else:
            # Self-attention on image features
            attended_features, _ = self.attention(
                spatial_features,
                spatial_features,
                spatial_features
            )
            # Average pool over spatial dimensions
            attended_features = attended_features.mean(dim=1)  # [batch, channels]
        
        # Project to output dimension
        features = self.spatial_projection(attended_features)
        
        return features


def create_vision_encoder(
    encoder_type: str = "cnn",
    backbone: str = "resnet50",
    feature_dim: int = 512,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create vision encoder
    
    Args:
        encoder_type: 'cnn' or 'attention'
        backbone: ResNet variant
        feature_dim: Output feature dimension
        pretrained: Use pre-trained weights
        **kwargs: Additional arguments
    
    Returns:
        Vision encoder model
    """
    if encoder_type == "cnn":
        return CNNVisionEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            **kwargs
        )
    elif encoder_type == "attention":
        return AttentionVisionEncoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test vision encoder
    print("Testing CNNVisionEncoder...")
    encoder = CNNVisionEncoder(backbone="resnet50", feature_dim=512)
    
    # Dummy batch
    batch_images = torch.randn(4, 3, 224, 224)
    
    features = encoder(batch_images)
    print(f"Input shape: {batch_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\nTesting AttentionVisionEncoder...")
    att_encoder = AttentionVisionEncoder(backbone="resnet50", feature_dim=512)
    
    features = att_encoder(batch_images)
    print(f"Output shape: {features.shape}")
    print(f"Parameters: {sum(p.numel() for p in att_encoder.parameters()):,}")
    
    print("\n  Vision encoder tests passed!")
