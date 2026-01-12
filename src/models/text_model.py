"""
Text-only models for VQA baseline
Implements LSTM-based question encoder for answering questions without images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTextModel(nn.Module):
    """
    LSTM-based text-only model for VQA
    Encodes questions and predicts answers without using images
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of answer classes
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Word embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> token
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)  # Zero out padding
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize classifier
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, questions: torch.Tensor):
        """
        Forward pass
        
        Args:
            questions: Tensor of shape (batch_size, seq_length) with token indices
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = questions.size(0)
        
        # Embed questions
        embedded = self.embedding(questions)  # (batch_size, seq_length, embedding_dim)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_length, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden_forward = hidden[-2]  # Last layer forward
            hidden_backward = hidden[-1]  # Last layer backward
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1]  # Last layer
        
        # Classify
        logits = self.classifier(final_hidden)  # (batch_size, num_classes)
        
        return logits
    
    def predict(self, questions: torch.Tensor):
        """
        Make predictions
        
        Args:
            questions: Tensor of shape (batch_size, seq_length)
            
        Returns:
            predictions: Tensor of shape (batch_size,) with predicted class indices
            probabilities: Tensor of shape (batch_size, num_classes) with class probabilities
        """
        logits = self.forward(questions)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities


class TransformerTextModel(nn.Module):
    """
    Transformer-based text-only model for VQA
    Uses transformer encoder instead of LSTM
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_length: int = 64
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of answer classes
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Positional embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.token_embedding.weight[0], 0)
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, questions: torch.Tensor):
        """
        Forward pass
        
        Args:
            questions: Tensor of shape (batch_size, seq_length)
            
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_length = questions.size()
        
        # Create position indices
        positions = torch.arange(seq_length, device=questions.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_emb = self.token_embedding(questions)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = self.dropout(token_emb + pos_emb)
        
        # Create padding mask (True for padding tokens)
        padding_mask = (questions == 0)
        
        # Pass through transformer
        encoded = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
        # Pool: use [CLS] token (first token) or mean pooling
        # Here we use mean pooling over non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = (encoded * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        pooled = sum_embeddings / count.clamp(min=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, questions: torch.Tensor):
        """Make predictions"""
        logits = self.forward(questions)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities


def create_text_model(
    model_type: str,
    vocab_size: int,
    num_classes: int,
    **kwargs
):
    """
    Factory function to create text models
    
    Args:
        model_type: 'lstm' or 'transformer'
        vocab_size: Vocabulary size
        num_classes: Number of answer classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        model: Initialized model
    """
    if model_type.lower() == 'lstm':
        model = LSTMTextModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type.lower() == 'transformer':
        model = TransformerTextModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
