"""
Multimodal VQA models combining text and vision
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .text_model import LSTMTextModel, TransformerTextModel
from .vision_encoder import CNNVisionEncoder, AttentionVisionEncoder


class MultimodalVQAModel(nn.Module):
    """
    Multimodal VQA model combining text and vision features
    """
    
    def __init__(
        self,
        # Text encoder params
        vocab_size: int,
        embedding_dim: int = 300,
        text_hidden_dim: int = 512,
        text_num_layers: int = 2,
        # Vision encoder params
        vision_backbone: str = "resnet50",
        vision_feature_dim: int = 512,
        vision_pretrained: bool = True,
        # Fusion params
        fusion_type: str = "concat",  # 'concat', 'attention', 'bilinear'
        fusion_hidden_dim: int = 512,
        # Output params
        num_classes: int = 4593,
        dropout: float = 0.3
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            text_hidden_dim: Hidden dimension for text encoder
            text_num_layers: Number of LSTM layers
            vision_backbone: ResNet variant for vision encoder
            vision_feature_dim: Vision feature dimension
            vision_pretrained: Use pretrained vision encoder
            fusion_type: Type of fusion ('concat', 'attention', 'bilinear')
            fusion_hidden_dim: Hidden dimension for fusion layer
            num_classes: Number of answer classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        
        # Text encoder (reuse from text baseline)
        self.text_encoder = LSTMTextModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            num_classes=num_classes,  # Temporary, won't use final layer
            dropout=dropout,
            bidirectional=True
        )
        
        # Remove text encoder's classifier (we'll use fusion classifier)
        text_output_dim = text_hidden_dim * 2  # bidirectional
        self.text_encoder.classifier = nn.Identity()
        
        # Vision encoder
        self.vision_encoder = CNNVisionEncoder(
            backbone=vision_backbone,
            feature_dim=vision_feature_dim,
            pretrained=vision_pretrained
        )
        
        # Fusion layer
        if fusion_type == "concat":
            # Simple concatenation
            fusion_input_dim = text_output_dim + vision_feature_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = fusion_hidden_dim
            
        elif fusion_type == "attention":
            # Attention-based fusion
            self.text_projection = nn.Linear(text_output_dim, fusion_hidden_dim)
            self.vision_projection = nn.Linear(vision_feature_dim, fusion_hidden_dim)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_hidden_dim,
                num_heads=8,
                batch_first=True
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = fusion_hidden_dim
            
        elif fusion_type == "bilinear":
            # Bilinear pooling
            self.bilinear = nn.Bilinear(text_output_dim, vision_feature_dim, fusion_hidden_dim)
            self.fusion = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = fusion_hidden_dim
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Final classifier
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
    
    def forward(
        self,
        questions: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            questions: Tensor of shape [batch_size, seq_len]
            images: Tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Encode text
        text_features = self._encode_text(questions)  # [batch, text_dim]
        
        # Encode vision
        vision_features = self.vision_encoder(images)  # [batch, vision_dim]
        
        # Fuse features
        if self.fusion_type == "concat":
            fused_features = self._concat_fusion(text_features, vision_features)
        elif self.fusion_type == "attention":
            fused_features = self._attention_fusion(text_features, vision_features)
        elif self.fusion_type == "bilinear":
            fused_features = self._bilinear_fusion(text_features, vision_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def _encode_text(self, questions: torch.Tensor) -> torch.Tensor:
        """Extract text features from questions"""
        # Get embeddings
        embedded = self.text_encoder.embedding(questions)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.text_encoder.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        if self.text_encoder.bidirectional:
            text_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            text_features = hidden[-1]
        
        # Apply dropout
        text_features = self.text_encoder.dropout(text_features)
        
        return text_features
    
    def _concat_fusion(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Simple concatenation fusion"""
        combined = torch.cat([text_features, vision_features], dim=1)
        fused = self.fusion(combined)
        return fused
    
    def _attention_fusion(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Attention-based fusion"""
        # Project to same dimension
        text_proj = self.text_projection(text_features).unsqueeze(1)  # [batch, 1, dim]
        vision_proj = self.vision_projection(vision_features).unsqueeze(1)  # [batch, 1, dim]
        
        # Cross-attention: text attends to vision
        attended_vision, _ = self.attention(text_proj, vision_proj, vision_proj)
        attended_vision = attended_vision.squeeze(1)  # [batch, dim]
        
        # Combine
        combined = torch.cat([text_proj.squeeze(1), attended_vision], dim=1)
        fused = self.fusion(combined)
        
        return fused
    
    def _bilinear_fusion(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear pooling fusion"""
        fused = self.bilinear(text_features, vision_features)
        fused = self.fusion(fused)
        return fused
    
    def predict(
        self,
        questions: torch.Tensor,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities
        
        Returns:
            predictions: Tensor of shape [batch_size]
            probabilities: Tensor of shape [batch_size, num_classes]
        """
        logits = self.forward(questions, images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        return predictions, probabilities


class CrossModalAttentionVQA(nn.Module):
    """
    Advanced VQA model with cross-modal attention
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        text_hidden_dim: int = 512,
        vision_feature_dim: int = 512,
        num_attention_heads: int = 8,
        num_classes: int = 4593,
        dropout: float = 0.3
    ):
        """Advanced cross-modal VQA with spatial attention"""
        super().__init__()
        
        self.num_classes = num_classes
        
        # Text encoder
        self.text_encoder = LSTMTextModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True
        )
        self.text_encoder.classifier = nn.Identity()
        text_output_dim = text_hidden_dim * 2
        
        # Vision encoder with attention
        self.vision_encoder = AttentionVisionEncoder(
            backbone="resnet50",
            feature_dim=vision_feature_dim,
            num_attention_heads=num_attention_heads,
            pretrained=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_feature_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Text projection to match vision dimension
        self.text_projection = nn.Linear(text_output_dim, vision_feature_dim)
        
        # Final fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(vision_feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(
        self,
        questions: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with cross-modal attention"""
        # Encode text
        embedded = self.text_encoder.embedding(questions)
        lstm_out, (hidden, cell) = self.text_encoder.lstm(embedded)
        text_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        text_features = self.text_encoder.dropout(text_features)
        
        # Project text to vision dimension
        text_proj = self.text_projection(text_features)
        
        # Encode vision with text-guided attention
        vision_features = self.vision_encoder(images, text_proj)
        
        # Cross-modal fusion
        text_query = text_proj.unsqueeze(1)  # [batch, 1, dim]
        vision_key = vision_features.unsqueeze(1)  # [batch, 1, dim]
        
        attended, _ = self.cross_attention(text_query, vision_key, vision_key)
        attended = attended.squeeze(1)
        
        # Combine and classify
        combined = torch.cat([text_proj, attended], dim=1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


def create_multimodal_model(
    model_type: str,
    vocab_size: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create multimodal VQA models
    
    Args:
        model_type: 'concat', 'attention', 'bilinear', 'cross_attention'
        vocab_size: Vocabulary size
        num_classes: Number of answer classes
        **kwargs: Additional model parameters
    
    Returns:
        Multimodal VQA model
    """
    if model_type in ["concat", "attention", "bilinear"]:
        return MultimodalVQAModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            fusion_type=model_type,
            **kwargs
        )
    elif model_type == "cross_attention":
        return CrossModalAttentionVQA(
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing MultimodalVQAModel...")
    
    # Test concat fusion
    model = create_multimodal_model(
        model_type="concat",
        vocab_size=10000,
        num_classes=4593,
        embedding_dim=128,
        text_hidden_dim=256,
        vision_feature_dim=512
    )
    
    # Dummy batch
    questions = torch.randint(0, 10000, (4, 32))
    images = torch.randn(4, 3, 224, 224)
    
    logits = model(questions, images)
    print(f"Input: questions{questions.shape}, images{images.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✓ Multimodal model tests passed!")
