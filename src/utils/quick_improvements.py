"""
Quick Improvement Implementation
Run this to create an improved training setup with key optimizations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from improved_training_config import (
    create_top_n_vocabulary, 
    get_enhanced_transforms,
    ImprovedMultimodalVQA,
    get_enhanced_training_config,
    create_enhanced_optimizer_scheduler,
    FocalLoss
)

def setup_improved_training():
    """Setup improved training with all optimizations"""
    
    project_root = Path(__file__).parent
    dataset_path = project_root / 'data'
    
    print("🚀 Setting up improved training configuration...")
    
    # 1. Create reduced vocabulary (1000 classes)
    print("\n1️⃣ Creating reduced vocabulary...")
    reduced_answers_file, reduced_answer_to_idx = create_top_n_vocabulary(
        train_csv_path=str(dataset_path / 'trainrenamed.csv'),
        answers_file_path=str(dataset_path / 'answers.txt'),
        top_n=1000
    )
    
    # 2. Get enhanced configuration
    config = get_enhanced_training_config()
    
    # 3. Setup transforms
    train_transforms, val_transforms = get_enhanced_transforms()
    
    print(f"\n2️⃣ Configuration loaded:")
    print(f"   - Classes reduced: 4,142 → 1,000")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['training']['base_lr']}")
    print(f"   - Dropout: {config['model']['dropout']}")
    
    return {
        'config': config,
        'reduced_answers_file': reduced_answers_file,
        'reduced_answer_to_idx': reduced_answer_to_idx,
        'train_transforms': train_transforms,
        'val_transforms': val_transforms
    }

def create_improved_model_and_training(vocab_size, num_classes, device):
    """Create improved model with enhanced training setup"""
    
    config = get_enhanced_training_config()
    
    # Create improved model
    model = ImprovedMultimodalVQA(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config['model']['embedding_dim'],
        text_hidden_dim=config['model']['text_hidden_dim'],
        fusion_hidden_dim=config['model']['fusion_hidden_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Create enhanced optimizer and scheduler
    optimizer, scheduler = create_enhanced_optimizer_scheduler(model, config)
    
    # Create focal loss
    criterion = FocalLoss(
        alpha=1, 
        gamma=2, 
        label_smoothing=config['training']['label_smoothing']
    )
    
    print(f"\n3️⃣ Model and training setup:")
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   - Vision LR: {config['training']['base_lr'] * config['training']['vision_lr_factor']:.2e}")
    print(f"   - Other LR: {config['training']['base_lr']:.2e}")
    print(f"   - Using Focal Loss with label smoothing")
    
    return model, optimizer, scheduler, criterion, config

# Example usage for notebook
def get_notebook_improvements():
    """Get improvements ready for notebook implementation"""
    
    improvements = setup_improved_training()
    
    print(f"\n🎯 Ready for notebook implementation!")
    print(f"\n📝 To use in your notebook:")
    print(f"1. Use reduced answers file: {Path(improvements['reduced_answers_file']).name}")
    print(f"2. Update batch_size to: {improvements['config']['training']['batch_size']}")
    print(f"3. Set dropout to: {improvements['config']['model']['dropout']}")
    print(f"4. Use enhanced data augmentation")
    print(f"5. Apply new learning rates and scheduler")
    
    return improvements

if __name__ == "__main__":
    # Run the setup
    improvements = get_notebook_improvements()
    
    print(f"\n✨ Expected improvements:")
    print(f"   - 🎯 Target accuracy: 35-45% (vs current 25%)")
    print(f"   - ⚡ Faster convergence with reduced classes")
    print(f"   - 🛡️ Better generalization with augmentation")
    print(f"   - 📈 More stable training with focal loss")
    
    print(f"\n📁 Files created:")
    print(f"   - improved_training_config.py")
    print(f"   - quick_improvements.py")
    print(f"   - {Path(improvements['reduced_answers_file']).name}")