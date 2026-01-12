"""
Quick test script to verify the setup works
Tests data loading and model creation without full training
"""
import torch
import yaml
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_text_dataloaders
from src.models.text_model import create_text_model

print("=" * 80)
print("Testing VQA Setup")
print("=" * 80)

# Load lightweight config
with open("config_lightweight.yaml", 'r') as f:
    config = yaml.safe_load(f)

print("\n✓ Configuration loaded")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")

# Test data loading with small batch
print("\n" + "-" * 80)
print("Testing Data Loading...")
print("-" * 80)

try:
    train_loader, val_loader, test_loader, vocab_size, num_classes, vocab = create_text_dataloaders(
        train_csv=config['data']['train_csv'],
        test_csv=config['data']['test_csv'],
        answers_file=config['data']['answers_file'],
        batch_size=4,  # Very small batch for testing
        val_split=config['data']['val_split'],
        num_workers=0,
        max_length=config['text']['max_length']
    )
    
    print(f"✓ Data loaders created successfully")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    
    # Test loading one batch
    batch = next(iter(train_loader))
    print(f"✓ Loaded sample batch")
    print(f"  - Question shape: {batch['question'].shape}")
    print(f"  - Answer shape: {batch['answer'].shape}")
    
except Exception as e:
    print(f"✗ Data loading failed: {str(e)}")
    sys.exit(1)

# Test model creation
print("\n" + "-" * 80)
print("Testing Model Creation...")
print("-" * 80)

try:
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Test forward pass
    model.to(device)
    questions = batch['question'].to(device)
    
    with torch.no_grad():
        outputs = model(questions)
    
    print(f"✓ Forward pass successful")
    print(f"  - Output shape: {outputs.shape}")
    
except Exception as e:
    print(f"✗ Model creation/forward pass failed: {str(e)}")
    sys.exit(1)

# Test training step
print("\n" + "-" * 80)
print("Testing Training Step...")
print("-" * 80)

try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    questions = batch['question'].to(device)
    answers = batch['answer'].to(device)
    
    # Forward pass
    outputs = model(questions)
    loss = criterion(outputs, answers)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == answers).float().mean()
    
    print(f"✓ Training step successful")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Batch accuracy: {accuracy.item():.4f}")
    
except Exception as e:
    print(f"✗ Training step failed: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour setup is ready for training.")
print("\nTo start lightweight training, run:")
print("  python train_text_model.py --config config_lightweight.yaml --model-type lstm")
print("\nFor even faster testing (just 2 epochs):")
print("  python quick_train.py")
