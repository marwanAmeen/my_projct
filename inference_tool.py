"""
VQA Inference Tool - Use trained models to answer questions about medical images

This tool loads your trained text-only and multimodal VQA models and allows
you to make predictions on new images and questions.

Usage:
    python inference_tool.py --image "path/to/image.png" --question "What is shown in the image?"
    python inference_tool.py --interactive  # Interactive mode
    python inference_tool.py --batch "path/to/test.csv"  # Batch processing
"""

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

# Import your models and utilities
from src.models.text_model import create_text_model
from src.models.multimodal_model import create_multimodal_model
from src.data.dataset import TextOnlyVQADataset, MultimodalVQADataset


class VQAInferenceTool:
    """Tool for making predictions with trained VQA models"""
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        text_checkpoint: str = "checkpoints/text_baseline_lstm_notebook/best_model.pth",
        multimodal_checkpoint: str = "checkpoints/multimodal_concat/best_model.pth",
        answers_file: str = "answers.txt",
        device: str = "auto"
    ):
        """
        Initialize the inference tool
        
        Args:
            config_path: Path to config.yaml
            text_checkpoint: Path to text-only model checkpoint
            multimodal_checkpoint: Path to multimodal model checkpoint  
            answers_file: Path to answers vocabulary file
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load answer vocabulary
        self.answers, self.answer_to_idx, self.idx_to_answer = self._load_answers(answers_file)
        self.num_classes = len(self.answers)
        
        # Build vocabulary (needed for text model)
        self.vocab, self.vocab_size = self._build_vocab()
        
        # Load models
        self.text_model = self._load_text_model(text_checkpoint)
        self.multimodal_model = self._load_multimodal_model(multimodal_checkpoint)
        
        # Setup image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ VQA Inference Tool ready!")
        print(f"  Device: {self.device}")
        print(f"  Text model loaded: {text_checkpoint}")
        print(f"  Multimodal model loaded: {multimodal_checkpoint}")
        print(f"  Answer vocabulary: {self.num_classes} classes")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _load_answers(self, answers_file: str) -> Tuple[List[str], Dict, Dict]:
        """Load answer vocabulary"""
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers = [line.strip() for line in f.readlines()]
        
        # Sort for consistency
        answers = sorted(list(set(answers)))
        answer_to_idx = {ans: idx for idx, ans in enumerate(answers)}
        idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}
        
        return answers, answer_to_idx, idx_to_answer
    
    def _build_vocab(self) -> Tuple[Dict, int]:
        """Build text vocabulary (simplified version for inference)"""
        # For inference, we'll build a basic vocab or load it from a saved file
        vocab_file = Path("vocab.json")
        
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
        else:
            # Basic vocabulary for inference
            vocab = {
                '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
                'what': 4, 'is': 5, 'the': 6, 'in': 7, 'of': 8, 'a': 9, 'and': 10,
                'to': 11, 'this': 12, 'image': 13, 'shown': 14, 'visible': 15
            }
        
        return vocab, len(vocab)
    
    def _load_text_model(self, checkpoint_path: str):
        """Load text-only VQA model"""
        model = create_text_model(
            vocab_size=self.vocab_size,
            embedding_dim=self.config['text']['embedding_dim'],
            hidden_dim=self.config['model']['baseline']['hidden_dim'],
            num_classes=self.num_classes,
            dropout=self.config['model']['baseline']['dropout']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _load_multimodal_model(self, checkpoint_path: str):
        """Load multimodal VQA model"""
        model = create_multimodal_model(
            model_type='concat',
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            embedding_dim=self.config['text']['embedding_dim'],
            text_hidden_dim=self.config['model']['baseline']['hidden_dim'],
            fusion_hidden_dim=self.config['model']['baseline']['hidden_dim'],
            dropout=self.config['model']['baseline']['dropout']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _encode_question(self, question: str, max_length: int = 32) -> torch.Tensor:
        """Encode question text to tensor"""
        words = question.lower().split()
        
        # Convert to indices
        indices = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices = indices + [self.vocab.get('<PAD>', 0)] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image
    
    def predict_single(
        self, 
        image_path: Optional[str] = None, 
        question: str = "",
        top_k: int = 5,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Make prediction for a single image-question pair
        
        Args:
            image_path: Path to image (None for text-only)
            question: Question text
            top_k: Return top-k predictions
            return_probabilities: Return prediction probabilities
        
        Returns:
            Dictionary with predictions and metadata
        """
        with torch.no_grad():
            # Encode question
            question_tensor = self._encode_question(question)
            
            # Text-only prediction
            text_logits = self.text_model(question_tensor)
            text_probs = F.softmax(text_logits, dim=1)
            text_top_k = torch.topk(text_probs, top_k, dim=1)
            
            text_predictions = [
                {
                    'answer': self.idx_to_answer[idx.item()],
                    'probability': prob.item(),
                    'confidence': prob.item()
                }
                for prob, idx in zip(text_top_k.values[0], text_top_k.indices[0])
            ]
            
            result = {
                'question': question,
                'text_only': {
                    'predictions': text_predictions,
                    'best_answer': text_predictions[0]['answer'],
                    'confidence': text_predictions[0]['confidence']
                }
            }
            
            # Multimodal prediction (if image provided)
            if image_path:
                image_tensor, original_image = self._load_and_preprocess_image(image_path)
                
                multimodal_logits = self.multimodal_model(question_tensor, image_tensor)
                multimodal_probs = F.softmax(multimodal_logits, dim=1)
                multimodal_top_k = torch.topk(multimodal_probs, top_k, dim=1)
                
                multimodal_predictions = [
                    {
                        'answer': self.idx_to_answer[idx.item()],
                        'probability': prob.item(),
                        'confidence': prob.item()
                    }
                    for prob, idx in zip(multimodal_top_k.values[0], multimodal_top_k.indices[0])
                ]
                
                result['multimodal'] = {
                    'predictions': multimodal_predictions,
                    'best_answer': multimodal_predictions[0]['answer'],
                    'confidence': multimodal_predictions[0]['confidence']
                }
                result['image_path'] = image_path
                result['original_image'] = original_image
        
        return result
    
    def predict_batch(self, csv_path: str, image_dir: str = "") -> pd.DataFrame:
        """
        Make predictions for a batch of image-question pairs
        
        Args:
            csv_path: Path to CSV with columns: image, question, (answer)
            image_dir: Directory containing images
        
        Returns:
            DataFrame with original data + predictions
        """
        df = pd.read_csv(csv_path)
        results = []
        
        print(f"Processing {len(df)} samples...")
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(df)}")
            
            image_path = None
            if 'image' in row and pd.notna(row['image']):
                image_path = Path(image_dir) / str(row['image'])
                if not image_path.suffix:
                    image_path = image_path.with_suffix('.png')
            
            prediction = self.predict_single(
                image_path=str(image_path) if image_path and image_path.exists() else None,
                question=str(row['question']),
                top_k=3
            )
            
            result_row = {
                'question': row['question'],
                'text_prediction': prediction['text_only']['best_answer'],
                'text_confidence': prediction['text_only']['confidence'],
            }
            
            if 'multimodal' in prediction:
                result_row['multimodal_prediction'] = prediction['multimodal']['best_answer']
                result_row['multimodal_confidence'] = prediction['multimodal']['confidence']
                result_row['image_path'] = str(image_path)
            
            if 'answer' in row:
                result_row['true_answer'] = row['answer']
                result_row['text_correct'] = (result_row['text_prediction'].lower() == 
                                            str(row['answer']).lower())
                if 'multimodal_prediction' in result_row:
                    result_row['multimodal_correct'] = (result_row['multimodal_prediction'].lower() == 
                                                      str(row['answer']).lower())
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def interactive_mode(self):
        """Interactive mode for testing predictions"""
        print("\n" + "="*60)
        print("VQA Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  'quit' or 'exit' - Exit interactive mode")
        print("  'help' - Show this help")
        print("  Just enter your question to continue")
        print("="*60)
        
        while True:
            try:
                question = input("\nEnter question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    break
                elif question.lower() == 'help':
                    print("Enter a medical question (optionally with image path)")
                    continue
                elif not question:
                    continue
                
                # Ask for image
                image_path = input("Enter image path (or press Enter to skip): ").strip()
                if not image_path or not Path(image_path).exists():
                    image_path = None
                    print("Using text-only prediction")
                
                # Make prediction
                result = self.predict_single(image_path, question, top_k=3)
                
                # Display results
                print(f"\nQuestion: {question}")
                if image_path:
                    print(f"Image: {image_path}")
                
                print(f"\nText-only prediction:")
                for i, pred in enumerate(result['text_only']['predictions'][:3]):
                    print(f"  {i+1}. {pred['answer']} ({pred['confidence']:.3f})")
                
                if 'multimodal' in result:
                    print(f"\nMultimodal prediction:")
                    for i, pred in enumerate(result['multimodal']['predictions'][:3]):
                        print(f"  {i+1}. {pred['answer']} ({pred['confidence']:.3f})")
                    
                    # Show comparison
                    text_answer = result['text_only']['best_answer']
                    mm_answer = result['multimodal']['best_answer']
                    if text_answer != mm_answer:
                        print(f"\n🔍 Models disagree!")
                        print(f"  Text-only: {text_answer}")
                        print(f"  Multimodal: {mm_answer}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    def compare_models(self, test_csv: str, image_dir: str = "") -> Dict:
        """Compare text-only vs multimodal model performance"""
        results_df = self.predict_batch(test_csv, image_dir)
        
        if 'true_answer' not in results_df.columns:
            print("No ground truth answers found - cannot compute accuracy")
            return results_df
        
        # Calculate accuracies
        text_accuracy = results_df['text_correct'].mean()
        
        comparison = {
            'text_only_accuracy': text_accuracy,
            'total_samples': len(results_df)
        }
        
        if 'multimodal_correct' in results_df.columns:
            mm_accuracy = results_df['multimodal_correct'].mean()
            comparison['multimodal_accuracy'] = mm_accuracy
            comparison['improvement'] = mm_accuracy - text_accuracy
            
            print(f"\nModel Comparison Results:")
            print(f"  Text-only accuracy: {text_accuracy:.4f} ({text_accuracy*100:.2f}%)")
            print(f"  Multimodal accuracy: {mm_accuracy:.4f} ({mm_accuracy*100:.2f}%)")
            print(f"  Improvement: {comparison['improvement']:.4f} ({comparison['improvement']*100:.2f} pp)")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description="VQA Inference Tool")
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--question', type=str, help='Question text')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--batch', type=str, help='Path to CSV file for batch processing')
    parser.add_argument('--image-dir', type=str, default='data/train', help='Directory containing images')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = VQAInferenceTool(config_path=args.config)
    
    if args.interactive:
        tool.interactive_mode()
    elif args.batch:
        print(f"Processing batch file: {args.batch}")
        results = tool.predict_batch(args.batch, args.image_dir)
        
        output_file = args.output or args.batch.replace('.csv', '_predictions.csv')
        results.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Show summary
        if 'true_answer' in results.columns:
            comparison = tool.compare_models(args.batch, args.image_dir)
    
    elif args.question:
        print(f"Making prediction...")
        result = tool.predict_single(args.image, args.question, top_k=5)
        
        print(f"\nQuestion: {args.question}")
        if args.image:
            print(f"Image: {args.image}")
        
        print(f"\nPredictions:")
        print(f"Text-only: {result['text_only']['best_answer']} ({result['text_only']['confidence']:.3f})")
        if 'multimodal' in result:
            print(f"Multimodal: {result['multimodal']['best_answer']} ({result['multimodal']['confidence']:.3f})")
    
    else:
        print("Please specify --interactive, --batch, or provide --question")
        print("Use --help for more options")


if __name__ == "__main__":
    main()