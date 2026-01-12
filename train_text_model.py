"""
Main training script for text-only VQA baseline
Run this to train the language model on questions and answers (without images)
"""
import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_text_dataloaders
from src.models.text_model import create_text_model
from src.training.trainer import TextVQATrainer


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function"""
    print("=" * 80)
    print("Text-Only VQA Baseline Training")
    print("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from: {args.config}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
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
        num_workers=0,  # Set to 0 for Windows
        max_length=config['text']['max_length']
    )
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of answer classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n" + "-" * 80)
    print("Creating Model...")
    print("-" * 80)
    
    # Create model
    model_type = args.model_type  # 'lstm' or 'transformer'
    
    if model_type == 'lstm':
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
    else:  # transformer
        model = create_text_model(
            model_type='transformer',
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            ff_dim=2048,
            dropout=0.1,
            max_length=config['text']['max_length']
        )
    
    print(f"Model type: {model_type.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n" + "-" * 80)
    print("Starting Training...")
    print("-" * 80)
    
    # Create trainer
    experiment_name = f"text_baseline_{model_type}"
    trainer = TextVQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=config['paths']['checkpoints'],
        experiment_name=experiment_name
    )
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    
    # Optionally evaluate on test set
    if args.eval_test:
        print("\n" + "-" * 80)
        print("Evaluating on Test Set...")
        print("-" * 80)
        
        # Load best model
        best_model_path = trainer.checkpoint_dir / "best_model.pth"
        trainer.load_checkpoint(best_model_path)
        
        # Create test trainer just for evaluation
        test_trainer = TextVQATrainer(
            model=model,
            train_loader=train_loader,  # Dummy
            val_loader=test_loader,
            config=config,
            device=device,
            checkpoint_dir=config['paths']['checkpoints'],
            experiment_name=experiment_name
        )
        
        test_metrics = test_trainer.validate()
        
        print(f"\nTest Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {test_metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"Precision (macro): {test_metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {test_metrics['recall_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text-only VQA model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "transformer"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on test set after training"
    )
    
    args = parser.parse_args()
    main(args)
