# VQA Model Inference Demo

**Interactive demo for using your trained VQA models**

This notebook provides easy-to-use functions for:
1. **Single predictions** - Test one image + question
2. **Batch processing** - Process multiple samples
3. **Model comparison** - Compare text-only vs multimodal results
4. **Visualization** - See predictions with images

## Setup

```python
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from torchvision import transforms

# Add project root to path
project_root = Path().absolute()
sys.path.insert(0, str(project_root))

# Import your models
from src.models.text_model import create_text_model
from src.models.multimodal_model import create_multimodal_model

print("✓ Setup complete!")
```

## Load Your Trained Models

```python
class VQAPredictor:
    """Easy-to-use VQA prediction interface"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load answer vocabulary
        with open('answers.txt', 'r', encoding='utf-8') as f:
            answers = [line.strip() for line in f.readlines()]
        
        self.answers = sorted(list(set(answers)))
        self.answer_to_idx = {ans: idx for idx, ans in enumerate(self.answers)}
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}
        self.num_classes = len(self.answers)
        
        # Simple vocab for inference
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
            'what': 4, 'is': 5, 'the': 6, 'in': 7, 'of': 8
        }
        self.vocab_size = len(self.vocab)
        
        # Load models
        self._load_models()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Models loaded on {self.device}")
        print(f"✓ {self.num_classes} answer classes")
    
    def _load_models(self):
        """Load both trained models"""
        # Text-only model
        self.text_model = create_text_model(
            vocab_size=self.vocab_size,
            embedding_dim=self.config['text']['embedding_dim'],
            hidden_dim=self.config['model']['baseline']['hidden_dim'],
            num_classes=self.num_classes,
            dropout=self.config['model']['baseline']['dropout']
        ).to(self.device)
        
        # Multimodal model  
        self.multimodal_model = create_multimodal_model(
            model_type='concat',
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            embedding_dim=self.config['text']['embedding_dim'],
            text_hidden_dim=self.config['model']['baseline']['hidden_dim'],
            fusion_hidden_dim=self.config['model']['baseline']['hidden_dim'],
            dropout=self.config['model']['baseline']['dropout']
        ).to(self.device)
        
        # Load checkpoints
        text_checkpoint = torch.load('checkpoints/text_baseline_lstm_notebook/best_model.pth', 
                                   map_location=self.device)
        multimodal_checkpoint = torch.load('checkpoints/multimodal_concat/best_model.pth',
                                         map_location=self.device)
        
        self.text_model.load_state_dict(text_checkpoint)
        self.multimodal_model.load_state_dict(multimodal_checkpoint)
        
        self.text_model.eval()
        self.multimodal_model.eval()
    
    def encode_question(self, question: str, max_length: int = 32):
        """Encode question to tensor"""
        words = question.lower().split()
        indices = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices = indices + [self.vocab.get('<PAD>', 0)] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict(self, image_path=None, question="", top_k=5, show_image=True):
        """Make prediction with both models"""
        with torch.no_grad():
            # Encode question
            question_tensor = self.encode_question(question)
            
            # Text-only prediction
            text_logits = self.text_model(question_tensor)
            text_probs = F.softmax(text_logits, dim=1)
            text_top_k = torch.topk(text_probs, top_k, dim=1)
            
            text_results = []
            for prob, idx in zip(text_top_k.values[0], text_top_k.indices[0]):
                text_results.append({
                    'answer': self.idx_to_answer[idx.item()],
                    'confidence': prob.item()
                })
            
            results = {
                'question': question,
                'text_only': text_results
            }
            
            # Multimodal prediction (if image provided)
            if image_path and Path(image_path).exists():
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
                
                # Make prediction
                mm_logits = self.multimodal_model(question_tensor, image_tensor)
                mm_probs = F.softmax(mm_logits, dim=1)
                mm_top_k = torch.topk(mm_probs, top_k, dim=1)
                
                mm_results = []
                for prob, idx in zip(mm_top_k.values[0], mm_top_k.indices[0]):
                    mm_results.append({
                        'answer': self.idx_to_answer[idx.item()],
                        'confidence': prob.item()
                    })
                
                results['multimodal'] = mm_results
                results['image'] = image
            
            return results

# Initialize predictor
predictor = VQAPredictor()
```

## 🎯 Quick Test - Single Prediction

```python
def quick_test(image_path, question):
    """Quick test function"""
    result = predictor.predict(image_path, question, top_k=3)
    
    # Display image if available
    if 'image' in result:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(result['image'])
        plt.axis('off')
        plt.title(f"Image: {Path(image_path).name}")
        
        # Display results
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        text = f"Question: {question}\\n\\n"
        text += "Text-only predictions:\\n"
        for i, pred in enumerate(result['text_only'][:3]):
            text += f"{i+1}. {pred['answer']} ({pred['confidence']:.3f})\\n"
        
        if 'multimodal' in result:
            text += "\\nMultimodal predictions:\\n"
            for i, pred in enumerate(result['multimodal'][:3]):
                text += f"{i+1}. {pred['answer']} ({pred['confidence']:.3f})\\n"
            
            # Highlight if different
            if result['text_only'][0]['answer'] != result['multimodal'][0]['answer']:
                text += "\\n🔍 Models give different answers!"
        
        plt.text(0.1, 0.5, text, fontsize=11, verticalalignment='center')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Question: {question}")
        print("\\nText-only predictions:")
        for i, pred in enumerate(result['text_only'][:3]):
            print(f"  {i+1}. {pred['answer']} ({pred['confidence']:.3f})")

# Example usage:
# quick_test("data/train/some_image.png", "What organ is shown?")
```

## 📊 Batch Evaluation

```python
def evaluate_on_test_set(test_csv="testrenamed.csv", image_dir="data/train", num_samples=100):
    """Evaluate both models on test set"""
    df = pd.read_csv(test_csv).head(num_samples)
    results = []
    
    print(f"Evaluating on {len(df)} test samples...")
    
    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(df)}")
        
        image_path = Path(image_dir) / f"{row['image']}.png"
        question = row['question']
        true_answer = row['answer']
        
        pred = predictor.predict(str(image_path), question, top_k=1)
        
        text_pred = pred['text_only'][0]['answer']
        text_conf = pred['text_only'][0]['confidence']
        
        result = {
            'question': question,
            'true_answer': true_answer,
            'text_prediction': text_pred,
            'text_confidence': text_conf,
            'text_correct': text_pred.lower() == true_answer.lower()
        }
        
        if 'multimodal' in pred:
            mm_pred = pred['multimodal'][0]['answer']
            mm_conf = pred['multimodal'][0]['confidence']
            result.update({
                'multimodal_prediction': mm_pred,
                'multimodal_confidence': mm_conf,
                'multimodal_correct': mm_pred.lower() == true_answer.lower()
            })
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Calculate accuracies
    text_acc = results_df['text_correct'].mean()
    print(f"\\nResults:")
    print(f"  Text-only accuracy: {text_acc:.4f} ({text_acc*100:.2f}%)")
    
    if 'multimodal_correct' in results_df.columns:
        mm_acc = results_df['multimodal_correct'].mean()
        improvement = mm_acc - text_acc
        print(f"  Multimodal accuracy: {mm_acc:.4f} ({mm_acc*100:.2f}%)")
        print(f"  Improvement: {improvement:.4f} ({improvement*100:.2f} pp)")
    
    return results_df

# Run evaluation
# results_df = evaluate_on_test_set(num_samples=200)
```

## 🔍 Analyze Specific Examples

```python
def show_examples(results_df, show_type='disagreement', num_examples=4):
    """Show specific examples"""
    
    if show_type == 'disagreement' and 'multimodal_prediction' in results_df.columns:
        # Show cases where models disagree
        disagreements = results_df[
            results_df['text_prediction'] != results_df['multimodal_prediction']
        ].sample(min(num_examples, len(results_df)))
        
        print("🔍 Cases where models disagree:")
        
        for idx, row in disagreements.iterrows():
            print(f"\\nQuestion: {row['question']}")
            print(f"True answer: {row['true_answer']}")
            print(f"Text-only: {row['text_prediction']} ({'✓' if row['text_correct'] else '✗'})")
            print(f"Multimodal: {row['multimodal_prediction']} ({'✓' if row['multimodal_correct'] else '✗'})")
            
    elif show_type == 'correct':
        correct_both = results_df[
            (results_df['text_correct'] == True) & 
            (results_df.get('multimodal_correct', True) == True)
        ].sample(min(num_examples, len(results_df)))
        
        print("✅ Cases where both models are correct:")
        for idx, row in correct_both.iterrows():
            print(f"Q: {row['question']}")
            print(f"A: {row['true_answer']} (both models got this right)")
            print()
    
    elif show_type == 'errors':
        errors = results_df[
            results_df['text_correct'] == False
        ].sample(min(num_examples, len(results_df)))
        
        print("❌ Error cases:")
        for idx, row in errors.iterrows():
            print(f"Q: {row['question']}")
            print(f"True: {row['true_answer']}")
            print(f"Predicted: {row['text_prediction']}")
            print()

# Usage examples:
# show_examples(results_df, 'disagreement', 5)
# show_examples(results_df, 'correct', 3)
# show_examples(results_df, 'errors', 3)
```

## 🎮 Interactive Testing

```python
def interactive_test():
    """Interactive testing interface"""
    print("🎮 Interactive VQA Testing")
    print("Commands: 'quit' to exit")
    
    while True:
        try:
            question = input("\\nEnter question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
                
            image_path = input("Image path (or Enter to skip): ").strip()
            if not image_path or not Path(image_path).exists():
                image_path = None
            
            quick_test(image_path, question)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Done!")

# Run interactive mode:
# interactive_test()
```

## 📈 Performance Comparison

```python
def plot_model_comparison(results_df):
    """Plot model performance comparison"""
    if 'multimodal_correct' not in results_df.columns:
        print("No multimodal results to compare")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    text_acc = results_df['text_correct'].mean()
    mm_acc = results_df['multimodal_correct'].mean()
    
    axes[0].bar(['Text-only', 'Multimodal'], [text_acc, mm_acc], 
               color=['skyblue', 'lightcoral'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylim(0, 1)
    
    # Add values on bars
    axes[0].text(0, text_acc + 0.02, f'{text_acc:.3f}', ha='center')
    axes[0].text(1, mm_acc + 0.02, f'{mm_acc:.3f}', ha='center')
    
    # Confidence distribution
    axes[1].hist(results_df['text_confidence'], alpha=0.5, label='Text-only', bins=20)
    axes[1].hist(results_df['multimodal_confidence'], alpha=0.5, label='Multimodal', bins=20)
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    
    # Agreement analysis
    agreement = (results_df['text_prediction'] == results_df['multimodal_prediction']).mean()
    axes[2].pie([agreement, 1-agreement], labels=['Agree', 'Disagree'], 
               autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
    axes[2].set_title('Model Agreement')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Model Agreement: {agreement:.3f} ({agreement*100:.1f}%)")

# Usage:
# plot_model_comparison(results_df)
```

---

## 🚀 Getting Started

1. **Run the setup cells** to load your models
2. **Try a quick test**: `quick_test("path/to/image.png", "What is shown?")`
3. **Evaluate on test set**: `results_df = evaluate_on_test_set()`
4. **Explore examples**: `show_examples(results_df, 'disagreement')`
5. **Interactive mode**: `interactive_test()`

Your models are trained and ready to use! 🎉