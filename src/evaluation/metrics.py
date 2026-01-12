"""
Evaluation metrics for VQA
"""
import torch
import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class VQAMetrics:
    """Metrics calculator for VQA tasks"""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of answer classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.question_texts = []
        self.answer_texts = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        question_texts: List[str] = None,
        answer_texts: List[str] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: Predicted class indices (batch_size,)
            targets: Ground truth class indices (batch_size,)
            question_texts: Optional list of question strings
            answer_texts: Optional list of answer strings
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
        
        if question_texts:
            self.question_texts.extend(question_texts)
        if answer_texts:
            self.answer_texts.extend(answer_texts)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary with metric names and values
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # F1 Score (macro and weighted)
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Precision and Recall
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
        
        # Exact match (for VQA)
        metrics['exact_match'] = np.mean(predictions == targets)
        
        return metrics
    
    def compute_per_question_type(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per question type (yes/no vs open-ended)
        
        Returns:
            Dictionary with metrics for each question type
        """
        if not self.answer_texts:
            return {}
        
        # Classify questions
        yes_no_indices = []
        open_indices = []
        
        for idx, answer in enumerate(self.answer_texts):
            answer_lower = answer.lower().strip()
            if answer_lower in ['yes', 'no']:
                yes_no_indices.append(idx)
            else:
                open_indices.append(idx)
        
        results = {}
        
        # Compute for yes/no questions
        if yes_no_indices:
            yn_preds = np.array([self.predictions[i] for i in yes_no_indices])
            yn_targets = np.array([self.targets[i] for i in yes_no_indices])
            results['yes_no'] = {
                'accuracy': accuracy_score(yn_targets, yn_preds),
                'count': len(yes_no_indices)
            }
        
        # Compute for open-ended questions
        if open_indices:
            open_preds = np.array([self.predictions[i] for i in open_indices])
            open_targets = np.array([self.targets[i] for i in open_indices])
            results['open_ended'] = {
                'accuracy': accuracy_score(open_targets, open_preds),
                'count': len(open_indices)
            }
        
        return results
    
    def get_confusion_stats(self, top_k: int = 10) -> Dict:
        """
        Get top confused answer pairs
        
        Args:
            top_k: Number of top confusions to return
            
        Returns:
            Dictionary with confusion statistics
        """
        confusion_counts = defaultdict(int)
        
        for pred, target in zip(self.predictions, self.targets):
            if pred != target:
                confusion_counts[(target, pred)] += 1
        
        # Sort by frequency
        top_confusions = sorted(
            confusion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return {
            'top_confusions': top_confusions,
            'total_errors': len([p for p, t in zip(self.predictions, self.targets) if p != t])
        }


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        
    Returns:
        Accuracy as float
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    return correct / total if total > 0 else 0.0


def calculate_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        logits: Model output logits
        targets: Ground truth class indices
        k: Top k predictions to consider
        
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        _, top_k_preds = logits.topk(k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()
        total = targets.size(0)
    
    return correct / total if total > 0 else 0.0
