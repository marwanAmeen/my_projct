"""Visualization utilities"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import torch
from PIL import Image


def plot_training_curves(train_losses: List[float], 
                         val_losses: List[float],
                         train_accs: Optional[List[float]] = None,
                         val_accs: Optional[List[float]] = None,
                         save_path: Optional[str] = None):
    """
    Plot training and validation curves
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accs: Training accuracies (optional)
        val_accs: Validation accuracies (optional)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12 if train_accs else 6, 4))
    
    if not train_accs:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    if train_accs and val_accs:
        axes[1].plot(train_accs, label='Train Acc', marker='o', markersize=3)
        axes[1].plot(val_accs, label='Val Acc', marker='s', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(images: List[torch.Tensor],
                         questions: List[str],
                         predictions: List[str],
                         ground_truths: List[str],
                         num_samples: int = 6,
                         save_path: Optional[str] = None):
    """
    Visualize model predictions
    
    Args:
        images: List of image tensors
        questions: List of questions
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        # Denormalize image
        img = images[idx].cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add text
        question = questions[idx][:50] + "..." if len(questions[idx]) > 50 else questions[idx]
        pred = predictions[idx][:30] + "..." if len(predictions[idx]) > 30 else predictions[idx]
        gt = ground_truths[idx][:30] + "..." if len(ground_truths[idx]) > 30 else ground_truths[idx]
        
        color = 'green' if predictions[idx] == ground_truths[idx] else 'red'
        
        title = f"Q: {question}\nPred: {pred}\nGT: {gt}"
        axes[idx].set_title(title, fontsize=9, color=color, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_answer_distribution(answers: List[str], 
                             top_n: int = 20,
                             save_path: Optional[str] = None):
    """
    Plot distribution of answers
    
    Args:
        answers: List of answers
        top_n: Number of top answers to plot
        save_path: Path to save the plot
    """
    from collections import Counter
    
    answer_counts = Counter(answers)
    top_answers = answer_counts.most_common(top_n)
    
    labels, counts = zip(*top_answers)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Answer')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Answers')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: List[str], 
                         y_pred: List[str],
                         labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix for classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
