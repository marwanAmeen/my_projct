"""
Quick Test Script - Compare Original vs Improved Models
This script loads both models and compares their performance on a small test batch
"""
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path().absolute()
sys.path.insert(0, str(project_root))

from src.data.dataset import create_multimodal_dataloaders
from src.models.multimodal_model import create_multimodal_model
from improved_multimodal_model import create_improved_model

def quick_comparison():
    """Quick comparison between original and improved models"""
    
    print("=" * 60)
    print("QUICK MODEL COMPARISON")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load small dataset for testing
    _, _, test_loader, vocab_size, num_classes, vocab, _ = create_multimodal_dataloaders(
        train_csv='data/trainrenamed.csv',
        test_csv='data/testrenamed.csv', 
        image_dir='data/train',
        answers_file='data/answers.txt',
        batch_size=8,  # Small batch for quick test
        val_split=0.1,
        num_workers=0,
        image_size=224
    )
    
    print(f" Test data loaded: {len(test_loader)} batches")
    
    # Create both models
    print("Creating models...")
    
    # Original model
    original_model = create_multimodal_model(
        model_type='concat',
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config['text']['embedding_dim'],
        text_hidden_dim=config['model']['baseline']['hidden_dim'],
        fusion_hidden_dim=config['model']['baseline']['hidden_dim'],
        dropout=config['model']['baseline']['dropout']
    ).to(device)
    
    # Improved model
    improved_model = create_improved_model(vocab_size, num_classes, config).to(device)
    
    # Compare model sizes
    orig_params = sum(p.numel() for p in original_model.parameters())
    impr_params = sum(p.numel() for p in improved_model.parameters())
    
    print(f" Original model: {orig_params:,} parameters ({orig_params * 4 / 1024**2:.1f} MB)")
    print(f" Improved model: {impr_params:,} parameters ({impr_params * 4 / 1024**2:.1f} MB)")
    print(f"  Size increase: {((impr_params/orig_params - 1) * 100):+.1f}%")
    
    # Test forward pass on both models
    print("\nTesting forward passes...")
    
    test_batch = next(iter(test_loader))
    questions = test_batch['question'].to(device)
    images = test_batch['image'].to(device)
    answers = test_batch['answer'].to(device)
    
    # Test original model
    original_model.eval()
    with torch.no_grad():
        orig_outputs = original_model(questions, images)
        orig_preds = torch.argmax(orig_outputs, dim=1)
        orig_acc = (orig_preds == answers).float().mean().item()
    
    # Test improved model  
    improved_model.eval()
    with torch.no_grad():
        impr_outputs = improved_model(questions, images)
        impr_preds = torch.argmax(impr_outputs, dim=1)
        impr_acc = (impr_preds == answers).float().mean().item()
    
    print(f"  Original model - Random accuracy on batch: {orig_acc:.4f} ({orig_acc*100:.2f}%)")
    print(f"  Improved model - Random accuracy on batch: {impr_acc:.4f} ({impr_acc*100:.2f}%)")
    
    # Compare output distributions
    orig_entropy = torch.distributions.Categorical(probs=torch.softmax(orig_outputs, dim=1)).entropy().mean().item()
    impr_entropy = torch.distributions.Categorical(probs=torch.softmax(impr_outputs, dim=1)).entropy().mean().item()
    
    print(f"  Original model - Output entropy: {orig_entropy:.4f}")
    print(f"  Improved model - Output entropy: {impr_entropy:.4f}")
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS IN NEW MODEL:")
    print("=" * 60)
    print("1.   Trainable vision encoder (vs frozen ResNet50)")
    print("2.   Spatial attention for vision features")  
    print("3.   Cross-modal attention fusion (vs simple concatenation)")
    print("4.   Differential learning rates for different components")
    print("5.   Enhanced regularization (dropout, label smoothing)")
    print("6.   Better weight initialization")
    print("7.   Gradient clipping and advanced scheduling")
    
    print("\nExpected improvements:")
    print("- Better domain adaptation to medical images")
    print("- Reduced overfitting with stronger regularization") 
    print("- More sophisticated multimodal fusion")
    print("- Improved training stability")
    

if __name__ == "__main__":
    quick_comparison()