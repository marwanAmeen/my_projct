"""
Improved Multimodal VQA Model with Better Fusion and Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedMultimodalVQA(nn.Module):
    """Improved multimodal VQA with cross-modal attention fusion"""
    
    def __init__(self, vocab_size, num_classes, embedding_dim=300, 
                 text_hidden_dim=512, fusion_hidden_dim=512, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # Text encoder (same as before)
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(embedding_dim, text_hidden_dim, 
                                batch_first=True, bidirectional=True)
        self.text_dropout = nn.Dropout(dropout)
        
        # Vision encoder (now trainable)
        self.vision_encoder = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.vision_encoder = nn.Sequential(*list(self.vision_encoder.children())[:-2])
        
        # Add spatial attention for vision
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Cross-modal fusion with attention
        self.vision_proj = nn.Linear(2048, fusion_hidden_dim)
        self.text_proj = nn.Linear(text_hidden_dim * 2, fusion_hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_hidden_dim, 
            num_heads=8, 
            dropout=dropout
        )
        
        # Final classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),  # Stronger dropout
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization"""
        for module in [self.text_embedding, self.vision_proj, self.text_proj, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, questions, images):
        batch_size = questions.size(0)
        
        # Text processing
        text_embedded = self.text_embedding(questions)
        text_output, (text_hidden, _) = self.text_lstm(text_embedded)
        
        # Use final hidden state (both directions)
        text_features = torch.cat([text_hidden[0], text_hidden[1]], dim=1)  # [batch, 1024]
        text_features = self.text_dropout(text_features)
        
        # Vision processing with spatial attention
        vision_maps = self.vision_encoder(images)  # [batch, 2048, 7, 7]
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(vision_maps)  # [batch, 1, 7, 7]
        attended_vision = vision_maps * attention_weights  # Broadcast multiply
        
        # Global average pooling
        vision_features = F.adaptive_avg_pool2d(attended_vision, 1).squeeze()  # [batch, 2048]
        
        # Project to same dimension
        vision_proj = self.vision_proj(vision_features)  # [batch, 512]
        text_proj = self.text_proj(text_features)        # [batch, 512]
        
        # Cross-modal attention: text queries vision
        text_proj = text_proj.unsqueeze(1)  # [batch, 1, 512]
        vision_proj = vision_proj.unsqueeze(1)  # [batch, 1, 512]
        
        # Attention: query=text, key=vision, value=vision
        attended_features, attention_weights = self.cross_attention(
            query=text_proj.transpose(0, 1),
            key=vision_proj.transpose(0, 1),
            value=vision_proj.transpose(0, 1)
        )
        
        # Back to [batch, hidden_dim]
        fused_features = attended_features.transpose(0, 1).squeeze(1)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits


class ImprovedMultimodalTrainer:
    """Enhanced trainer with better optimization strategies"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Enhanced optimizer with differential learning rates
        vision_params = list(model.vision_encoder.parameters()) + \
                      list(model.spatial_attention.parameters()) + \
                      list(model.vision_proj.parameters())
        
        text_params = list(model.text_embedding.parameters()) + \
                     list(model.text_lstm.parameters()) + \
                     list(model.text_proj.parameters())
        
        fusion_params = list(model.cross_attention.parameters()) + \
                       list(model.classifier.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': vision_params, 'lr': config['training']['learning_rate'] * 0.1},  # Lower LR
            {'params': text_params, 'lr': config['training']['learning_rate']},
            {'params': fusion_params, 'lr': config['training']['learning_rate']}
        ], weight_decay=config['training']['weight_decay'])
        
        # Enhanced loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Enhanced scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            questions = batch['question'].to(self.device)
            images = batch['image'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Forward pass
            logits = self.model(questions, images)
            loss = self.criterion(logits, answers)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()
        
        return total_loss / len(self.train_loader), correct / total


def create_improved_model(vocab_size, num_classes, config):
    """Create improved multimodal model"""
    model = ImprovedMultimodalVQA(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config['text']['embedding_dim'],
        text_hidden_dim=config['model']['baseline']['hidden_dim'],
        fusion_hidden_dim=config['model']['baseline']['hidden_dim'],
        dropout=0.5  # Increased dropout
    )
    return model


# Configuration updates for better performance
IMPROVED_CONFIG = {
    'training': {
        'batch_size': 12,  # Slightly smaller due to more complex model
        'num_epochs': 15,  # More epochs
        'learning_rate': 5e-5,  # Lower learning rate
        'weight_decay': 1e-3,  # Stronger regularization
        'early_stopping_patience': 7,  # More patience
        'gradient_clip': 1.0
    },
    'text': {
        'embedding_dim': 300,
        'max_length': 20
    },
    'model': {
        'baseline': {
            'hidden_dim': 512,
            'dropout': 0.5  # Increased
        }
    }
}