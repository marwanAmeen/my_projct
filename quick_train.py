"""
Quick training script for testing - runs only 2 epochs
Perfect for low-spec laptops to verify everything works
"""
import torch
import yaml
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_text_dataloaders
from src.models.text_model import create_text_model
from src.training.trainer import TextVQATrainer

print("=" * 80)
print("Quick Training Test (2 Epochs)")
print("=" * 80)

# Load lightweight config
with open("config_lightweight.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Override for quick test
config['training']['num_epochs'] = 2
config['training']['batch_size'] = 8
config['training']['early_stopping_patience'] = 10  # Disable early stopping

print(f"\n✓ Using lightweight configuration")
print(f"  - Batch size: {config['training']['batch_size']}")
print(f"  - Epochs: {config['training']['num_epochs']}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

# Set seed
torch.manual_seed(config['seed'])

print("\n" + "-" * 80)
print("Loading Data...")
print("-" * 80)

# Create dataloaders
train_loader, val_loader, test_loader, vocab_size, num_classes, vocab = create_text_dataloaders(
    train_csv=config['data']['train_csv'],
    test_csv=config['data']['test_csv'],
    answers_file=config['data']['answers_file'],
    batch_size=config['training']['batch_size'],
    val_split=config['data']['val_split'],
    num_workers=0,
    max_length=config['text']['max_length']
)

print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

print("\n" + "-" * 80)
print("Creating Model...")
print("-" * 80)

# Create model
model = create_text_model(
    model_type='lstm',
    vocab_size=vocab_size,
    num_classes=num_classes,
    embedding_dim=config['text']['embedding_dim'],
    hidden_dim=config['model']['baseline']['hidden_dim'],
    num_layers=config['model']['baseline']['num_layers'],
    dropout=config['model']['baseline']['dropout'],
    bidirectional=True
)

print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

print("\n" + "-" * 80)
print("Starting Training...")
print("-" * 80)

# Create trainer
trainer = TextVQATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    checkpoint_dir=config['paths']['checkpoints'],
    experiment_name="quick_test"
)

# Train
trainer.train()

print("\n" + "=" * 80)
print("Quick Training Complete!")
print("=" * 80)
print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
print("\nNow you can run full training with:")
print("  python train_text_model.py --config config_lightweight.yaml --model-type lstm")
