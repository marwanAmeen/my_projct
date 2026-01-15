"""
Quick VQA Test Script
Test your trained models with a simple example
"""

import torch
import torch.nn.functional as F
from PIL import Image
import yaml
from pathlib import Path
from torchvision import transforms
import sys

# Add project root
sys.path.insert(0, str(Path().absolute()))

from src.models.text_model import create_text_model
from src.models.multimodal_model import create_multimodal_model

def load_models():
    """Load trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load answers
    with open('answers.txt', 'r', encoding='utf-8') as f:
        answers = sorted([line.strip() for line in f.readlines()])
    
    idx_to_answer = {idx: ans for idx, ans in enumerate(answers)}
    num_classes = len(answers)
    
    # Simple vocab for testing
    vocab = {'<PAD>': 0, '<UNK>': 1, 'what': 4, 'is': 5, 'the': 6, 'in': 7, 'of': 8}
    vocab_size = len(vocab)
    
    # Load text model
    text_model = create_text_model(
        vocab_size=vocab_size,
        embedding_dim=config['text']['embedding_dim'],
        hidden_dim=config['model']['baseline']['hidden_dim'],
        num_classes=num_classes,
        dropout=config['model']['baseline']['dropout']
    ).to(device)
    
    # Load multimodal model
    multimodal_model = create_multimodal_model(
        model_type='concat',
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=config['text']['embedding_dim'],
        text_hidden_dim=config['model']['baseline']['hidden_dim'],
        fusion_hidden_dim=config['model']['baseline']['hidden_dim'],
        dropout=config['model']['baseline']['dropout']
    ).to(device)
    
    # Load checkpoints
    text_checkpoint = torch.load('checkpoints/text_baseline_lstm_notebook/best_model.pth', 
                               map_location=device)
    multimodal_checkpoint = torch.load('checkpoints/multimodal_concat/best_model.pth',
                                     map_location=device)
    
    text_model.load_state_dict(text_checkpoint)
    multimodal_model.load_state_dict(multimodal_checkpoint)
    
    text_model.eval()
    multimodal_model.eval()
    
    return text_model, multimodal_model, vocab, idx_to_answer, device

def encode_question(question, vocab, device, max_length=32):
    """Encode question"""
    words = question.lower().split()
    indices = [vocab.get(word, vocab.get('<UNK>', 1)) for word in words]
    
    if len(indices) < max_length:
        indices = indices + [vocab.get('<PAD>', 0)] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

def test_models(question="What is shown in the image?", image_path=None):
    """Test both models"""
    print("Loading models...")
    text_model, multimodal_model, vocab, idx_to_answer, device = load_models()
    
    print(f"✓ Models loaded on {device}")
    print(f"✓ Answer vocabulary: {len(idx_to_answer)} classes")
    
    # Encode question
    question_tensor = encode_question(question, vocab, device)
    
    with torch.no_grad():
        # Text-only prediction
        text_logits = text_model(question_tensor)
        text_probs = F.softmax(text_logits, dim=1)
        text_top3 = torch.topk(text_probs, 3, dim=1)
        
        print(f"\nQuestion: {question}")
        print(f"Text-only predictions:")
        for i, (prob, idx) in enumerate(zip(text_top3.values[0], text_top3.indices[0])):
            answer = idx_to_answer[idx.item()]
            confidence = prob.item()
            print(f"  {i+1}. {answer} ({confidence:.3f})")
        
        # Multimodal prediction (if image provided)
        if image_path and Path(image_path).exists():
            # Load image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            mm_logits = multimodal_model(question_tensor, image_tensor)
            mm_probs = F.softmax(mm_logits, dim=1)
            mm_top3 = torch.topk(mm_probs, 3, dim=1)
            
            print(f"\nMultimodal predictions (with image):")
            for i, (prob, idx) in enumerate(zip(mm_top3.values[0], mm_top3.indices[0])):
                answer = idx_to_answer[idx.item()]
                confidence = prob.item()
                print(f"  {i+1}. {answer} ({confidence:.3f})")
            
            # Compare top predictions
            text_answer = idx_to_answer[text_top3.indices[0][0].item()]
            mm_answer = idx_to_answer[mm_top3.indices[0][0].item()]
            
            if text_answer != mm_answer:
                print(f"\n🔍 Models disagree!")
                print(f"  Text-only: {text_answer}")
                print(f"  Multimodal: {mm_answer}")
            else:
                print(f"\n✅ Both models agree: {text_answer}")
        
        else:
            print("\nNo image provided - only text-only prediction shown")

if __name__ == "__main__":
    # Example usage
    print("🎯 VQA Model Test")
    print("=" * 50)
    
    # Test with text only
    test_models("What organ is visible?")
    
    print("\n" + "=" * 50)
    
    # Test with image (if available)
    # Uncomment and modify path to test with an image:
    # test_models("What abnormality is shown?", "data/train/example.png")
    
    print("\nTo test with your own examples:")
    print("1. Modify the question in the script")
    print("2. Add an image path to test multimodal prediction")
    print("3. Run: python test_models.py")