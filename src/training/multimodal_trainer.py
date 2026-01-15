"""
Training module for multimodal VQA models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict

from ..evaluation.metrics import VQAMetrics, AverageMeter, calculate_accuracy


class MultimodalVQATrainer:
    """Trainer for multimodal VQA models (text + vision)"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "multimodal_vqa"
    ):
        """
        Args:
            model: Multimodal VQA model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Training history for plotting
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Metrics
        self.train_metrics = VQAMetrics(model.num_classes)
        self.val_metrics = VQAMetrics(model.num_classes)
        
        print(f"Initialized multimodal trainer for {experiment_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        num_epochs = self.config['training']['num_epochs']
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3
            )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            questions = batch['question'].to(self.device)
            images = batch['image'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Forward pass
            logits = self.model(questions, images)
            loss = self.criterion(logits, answers)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            predictions = logits.argmax(dim=1)
            batch_acc = calculate_accuracy(predictions, answers)
            
            # Update meters
            loss_meter.update(loss.item(), questions.size(0))
            acc_meter.update(batch_acc, questions.size(0))
            
            # Update metrics
            self.train_metrics.update(
                predictions,
                answers,
                batch.get('question_text'),
                batch.get('answer_text')
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                questions = batch['question'].to(self.device)
                images = batch['image'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                # Forward pass
                logits = self.model(questions, images)
                loss = self.criterion(logits, answers)
                
                # Calculate metrics
                predictions = logits.argmax(dim=1)
                batch_acc = calculate_accuracy(predictions, answers)
                
                # Update meters
                loss_meter.update(loss.item(), questions.size(0))
                acc_meter.update(batch_acc, questions.size(0))
                
                # Update metrics
                self.val_metrics.update(
                    predictions,
                    answers,
                    batch.get('question_text'),
                    batch.get('answer_text')
                )
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.4f}'
                })
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def save_checkpoint(self, filename: str = "checkpoint.pth", is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def get_history(self):
        """Get training history for plotting"""
        return {
            'train_loss': self.train_losses.copy(),
            'train_acc': self.train_accuracies.copy(),
            'val_loss': self.val_losses.copy(),
            'val_acc': self.val_accuracies.copy()
        }
    
    def reset_history(self):
        """Reset training history"""
        self.train_losses.clear()
        self.train_accuracies.clear()
        self.val_losses.clear()
        self.val_accuracies.clear()
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']
        
        print(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            else:
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            
            # Store training history
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Also store in training_history dict for compatibility
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Check if best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
    
    def evaluate(self, test_loader: DataLoader):
        """Evaluate model on test set"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(test_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch in pbar:
                questions = batch['question'].to(self.device)
                images = batch['image'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                # Forward pass
                logits = self.model(questions, images)
                loss = self.criterion(logits, answers)
                
                # Calculate predictions
                predictions = logits.argmax(dim=1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(answers.cpu().numpy().tolist())
                
                # Update loss meter
                loss_meter.update(loss.item(), questions.size(0))
                
                # Calculate batch accuracy for progress bar
                batch_acc = calculate_accuracy(predictions, answers)
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
        
        # Calculate final accuracy
        total_correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        test_acc = total_correct / len(all_predictions)
        
        return loss_meter.avg, test_acc, all_predictions, all_labels
    
    @property
    def history(self):
        """Property to access training history"""
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies, 
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies
        }
