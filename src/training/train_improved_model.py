"""
Quick Training Script for Improved Multimodal Model
This script integrates the improved model with  existing data pipeline
"""
import sys
import os
from pathlib import Path
import torch
import yaml
import time
from tqdm import tqdm

def quick_train_improved_model(project_root=None):
    """Quick training function for the improved multimodal model"""
    
    # Set default project root if not provided
    if project_root is None:
        project_root = Path().absolute()
    else:
        project_root = Path(project_root)
    
    # Add project root to path for imports
    sys.path.insert(0, str(project_root))
    
    # Import your existing modules (after path is set)
    from src.data.dataset import create_multimodal_dataloaders
    from src.evaluation.metrics import calculate_accuracy
    from improved_multimodal_model import create_improved_model, ImprovedMultimodalTrainer
    
    print("=" * 60)
    print("TRAINING IMPROVED MULTIMODAL MODEL")
    print("=" * 60)
    print(f"Using project root: {project_root}")
    
    # 1. Load configuration
    config_path = project_root / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for improved model
    config['training'].update({
        'batch_size': 12,  # Smaller due to more complex model
        'num_epochs': 15,  # More epochs
        'learning_rate': 5e-5,  # Lower learning rate
        'weight_decay': 1e-3,  # Stronger regularization
        'early_stopping_patience': 7
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load data (using your existing data loader)
    print("Loading data...")
    
    # Updated paths to use data/ folder structure
    data_dir = project_root / 'data'
    dataset_path = data_dir / 'train'  # Images in data/train folder
    
    train_loader, val_loader, test_loader, vocab_size, num_classes, vocab, answer_to_idx = create_multimodal_dataloaders(
        train_csv=str(data_dir / 'trainrenamed.csv'),
        test_csv=str(data_dir / 'testrenamed.csv'),
        image_dir=str(dataset_path),
        answers_file=str(data_dir / 'answers.txt'),
        batch_size=config['training']['batch_size'],
        val_split=0.1,
        num_workers=0,
        image_size=224
    )
    
    print(f"  Data loaded: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")
    print(f"  Vocabulary size: {vocab_size}, Classes: {num_classes}")
    
    # 3. Create improved model
    print("Creating improved multimodal model...")
    model = create_improved_model(vocab_size, num_classes, config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    # 4. Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        test_questions = test_batch['question'].to(device)
        test_images = test_batch['image'].to(device)
        test_answers = test_batch['answer'].to(device)
        
        outputs = model(test_questions, test_images)
        print(f"  Forward pass successful: {outputs.shape}")
    
    # 5. Create enhanced trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=project_root / 'checkpoints' / 'improved_multimodal'
    )
    
    print(f"  Trainer initialized with differential learning rates")
    
    # 6. Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch()
        
        # Validate
        val_loss, val_acc = trainer.validate()
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            trainer.save_checkpoint(is_best=True)
            print(f" New best model saved! Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # 7. Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_best_checkpoint()
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("IMPROVED MODEL TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nComparison:")
    print(f"  Text-only baseline: 47.36%")
    print(f"  Original multimodal: 41.25%")
    print(f"  Improved multimodal: {test_acc*100:.2f}%")
    
    improvement_vs_text = (test_acc - 0.4736) * 100
    improvement_vs_original = (test_acc - 0.4125) * 100
    
    print(f"  Improvement vs text baseline: {improvement_vs_text:+.2f} pp")
    print(f"  Improvement vs original multimodal: {improvement_vs_original:+.2f} pp")
    
    return {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'training_time': training_time,
        'total_epochs': epoch + 1
    }


class EnhancedTrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, model, train_loader, val_loader, config, device, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced optimizer with differential learning rates
        vision_params = []
        text_params = []
        fusion_params = []
        
        # Categorize parameters
        for name, param in model.named_parameters():
            if 'vision_encoder' in name or 'spatial_attention' in name or 'vision_proj' in name:
                vision_params.append(param)
            elif 'text_' in name or 'embedding' in name:
                text_params.append(param)
            else:
                fusion_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': vision_params, 'lr': config['training']['learning_rate'] * 0.1, 'name': 'vision'},
            {'params': text_params, 'lr': config['training']['learning_rate'], 'name': 'text'},
            {'params': fusion_params, 'lr': config['training']['learning_rate'], 'name': 'fusion'}
        ], weight_decay=config['training']['weight_decay'])
        
        # Enhanced loss with label smoothing
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Enhanced scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
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
            
            # Update progress bar
            current_acc = correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                questions = batch['question'].to(self.device)
                images = batch['image'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                logits = self.model(questions, images)
                loss = self.criterion(logits, answers)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += answers.size(0)
                correct += predicted.eq(answers).sum().item()
        
        return total_loss / len(self.val_loader), correct / total
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                questions = batch['question'].to(self.device)
                images = batch['image'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                logits = self.model(questions, images)
                loss = self.criterion(logits, answers)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += answers.size(0)
                correct += predicted.eq(answers).sum().item()
        
        return total_loss / len(test_loader), correct / total
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_improved_model.pth')
    
    def load_best_checkpoint(self):
        checkpoint_path = self.checkpoint_dir / 'best_improved_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded best checkpoint from {checkpoint_path}")
        else:
            print("   No checkpoint found, using current model state")


if __name__ == "__main__":
    import sys
    
    print("Starting improved multimodal VQA training...")
    
    # Allow passing project root as command line argument
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
        print(f"Using provided project root: {project_root}")
    else:
        project_root = Path().absolute()
        print(f"Using current directory as project root: {project_root}")
    
    results = quick_train_improved_model(project_root)
    print(f"\nFinal Results: {results}")