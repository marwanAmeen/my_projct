"""Dataset classes for PathVQA"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Optional, Tuple
import numpy as np


def build_answer_vocab(answers):
    """Build answer vocabulary with consistent indexing"""
    # Get unique answers and sort them for consistent indexing
    unique_answers = sorted(list(set(answers)))

    # Create mapping for unique answers
    ans_to_idx = {ans: i for i, ans in enumerate(unique_answers)}

    # Add a special token for unknown answers if it's not already present
    # It's crucial that this token, if used, gets an index within the valid range.
    # Its index will be the last one.
    if '<UNK>' not in ans_to_idx:
        ans_to_idx['<UNK>'] = len(ans_to_idx)

    idx_to_ans = {i: ans for ans, i in ans_to_idx.items()}
    return ans_to_idx, idx_to_ans


class PathVQADataset(Dataset):
    """PathVQA Dataset for Medical Visual Question Answering"""
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        answers_file: Optional[str] = None,
        transform=None,
        tokenizer=None,
        max_length: int = 64,
        answer_to_idx: Optional[Dict] = None,
    ):
        """
        Args:
            csv_file: Path to CSV file with image, question, answer columns
            image_dir: Directory with all the images
            answers_file: Path to file with unique answers (for building vocabulary)
            transform: Optional transform to be applied on images
            tokenizer: Tokenizer for questions (if None, uses simple word tokenization)
            max_length: Maximum length for question tokens
            answer_to_idx: Dictionary mapping answers to indices (if None, builds from data)
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build answer vocabulary
        if answer_to_idx is None:
            if answers_file and os.path.exists(answers_file):
                with open(answers_file, 'r', encoding='utf-8') as f:
                    answers = [line.strip() for line in f.readlines()]
            else:
                answers = self.data['answer'].unique().tolist()
            
            # Use the proper answer vocabulary builder
            self.answer_to_idx, self.idx_to_answer = build_answer_vocab(answers)
        else:
            self.answer_to_idx = answer_to_idx
            self.idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}
        
        self.num_classes = len(self.answer_to_idx)
        
        print(f"Loaded {len(self.data)} samples with {self.num_classes} unique answers")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - question: Tokenized question
                - answer: Answer index
                - answer_text: Original answer text
        """
        row = self.data.iloc[idx]
        
        # Load image
        image_name = row['image']
        if not image_name.endswith('.png'):
            image_name = f"{image_name}.png"
        
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process question
        question = row['question']
        if self.tokenizer:
            # Use provided tokenizer (e.g., from transformers)
            question_tokens = self.tokenizer(
                question,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            question_encoding = {
                'input_ids': question_tokens['input_ids'].squeeze(0),
                'attention_mask': question_tokens['attention_mask'].squeeze(0)
            }
        else:
            # Simple word tokenization (for baseline)
            question_encoding = question
        
        # Process answer
        answer_text = row['answer']
        answer_idx = self.answer_to_idx.get(answer_text, self.answer_to_idx.get('<UNK>', 0))  # Use <UNK> token if unknown
        
        return {
            'image': image,
            'question': question_encoding,
            'question_text': question,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'answer_text': answer_text,
            'image_name': image_name
        }
    
    def get_answer_text(self, idx: int) -> str:
        """Get answer text from index"""
        return self.idx_to_answer.get(idx, "<unknown>")


class VQACollator:
    """Custom collator for batching VQA samples"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        """Collate batch of samples"""
        images = torch.stack([item['image'] for item in batch])
        answers = torch.stack([item['answer'] for item in batch])
        
        if self.tokenizer:
            # If using transformer tokenizer
            questions = {
                'input_ids': torch.stack([item['question']['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['question']['attention_mask'] for item in batch])
            }
        else:
            # For simple tokenization
            questions = [item['question'] for item in batch]
        
        question_texts = [item['question_text'] for item in batch]
        answer_texts = [item['answer_text'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        
        return {
            'images': images,
            'questions': questions,
            'question_texts': question_texts,
            'answers': answers,
            'answer_texts': answer_texts,
            'image_names': image_names
        }


class TextOnlyVQADataset(Dataset):
    """
    Text-only VQA Dataset for language model baseline
    This dataset only processes questions and answers without images
    """
    
    def __init__(
        self,
        csv_file: str,
        answers_file: str,
        vocab: Optional[Dict] = None,
        max_length: int = 64,
        mode: str = 'train'
    ):
        """
        Args:
            csv_file: Path to CSV with columns: image, question, answer
            answers_file: Path to text file with all possible answers
            vocab: Pre-built vocabulary (None to build from data)
            max_length: Maximum sequence length for questions
            mode: 'train', 'val', or 'test'
        """
        self.mode = mode
        self.max_length = max_length
        
        # Load data
        self.data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        
        # Load answer vocabulary
        with open(answers_file, 'r', encoding='utf-8') as f:
            self.answers = [line.strip() for line in f.readlines()]
        
        # Create answer to index mapping
        self.answer_to_idx = {answer: idx for idx, answer in enumerate(self.answers)}
        self.idx_to_answer = {idx: answer for idx, answer in enumerate(self.answers)}
        self.num_classes = len(self.answers)
        
        print(f"Answer vocabulary size: {self.num_classes}")
        
        # Build or use vocabulary
        if vocab is None:
            self._build_vocabulary()
        else:
            self.vocab = vocab
            self.vocab_size = len(vocab)
    
    def _build_vocabulary(self):
        """Build vocabulary from questions"""
        from collections import Counter
        
        # Tokenize all questions
        all_tokens = []
        for question in self.data['question']:
            tokens = self._simple_tokenize(question)
            all_tokens.extend(tokens)
        
        # Count tokens and take most common
        token_counts = Counter(all_tokens)
        vocab_size = 10000  # Limit vocabulary size
        most_common = token_counts.most_common(vocab_size - 4)  # Reserve 4 special tokens
        
        # Create vocabulary
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        for idx, (token, _) in enumerate(most_common, start=4):
            self.vocab[token] = idx
        
        self.vocab_size = len(self.vocab)
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def _simple_tokenize(self, text: str):
        """Simple word tokenization"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        tokens = text.split()
        return tokens
    
    def _encode_question(self, question: str) -> torch.Tensor:
        """Encode question to token indices"""
        tokens = self._simple_tokenize(question)
        
        # Convert to indices
        indices = [self.vocab['<SOS>']]
        for token in tokens[:self.max_length - 2]:  # Reserve space for SOS/EOS
            indices.append(self.vocab.get(token, self.vocab['<UNK>']))
        indices.append(self.vocab['<EOS>'])
        
        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.vocab['<PAD>'])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _encode_answer(self, answer: str) -> int:
        """Encode answer to class index"""
        answer = answer.strip().lower()
        
        # Try to find exact match
        if answer in self.answer_to_idx:
            return self.answer_to_idx[answer]
        
        # Try to find case-insensitive match
        for ans, idx in self.answer_to_idx.items():
            if ans.lower() == answer:
                return idx
        
        # Default to first class if not found
        return 0
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary with:
                - question: Encoded question tensor [max_length]
                - answer: Answer class index
                - question_text: Original question string
                - answer_text: Original answer string
        """
        row = self.data.iloc[idx]
        
        question = row['question']
        answer = row['answer']
        
        # Encode question
        question_encoded = self._encode_question(question)
        
        # Encode answer
        answer_idx = self._encode_answer(answer)
        
        return {
            'question': question_encoded,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'question_text': question,
            'answer_text': answer
        }
    
    def decode_question(self, encoded: torch.Tensor) -> str:
        """Decode token indices back to text"""
        if not hasattr(self, 'idx_to_vocab'):
            self.idx_to_vocab = {idx: token for token, idx in self.vocab.items()}
        
        tokens = []
        for idx in encoded.tolist():
            token = self.idx_to_vocab.get(idx, '<UNK>')
            if token in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(token)
        
        return ' '.join(tokens)


def create_text_dataloaders(
    train_csv: str,
    test_csv: str,
    answers_file: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 0,
    max_length: int = 64
):
    """
    Create train, validation, and test dataloaders for text-only training
    
    Returns:
        train_loader, val_loader, test_loader, vocab_size, num_classes, vocab
    """
    from torch.utils.data import DataLoader, random_split
    
    # Load full training dataset
    full_train_dataset = TextOnlyVQADataset(
        csv_file=train_csv,
        answers_file=answers_file,
        max_length=max_length,
        mode='train'
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create test dataset (shares vocabulary with train)
    test_dataset = TextOnlyVQADataset(
        csv_file=test_csv,
        answers_file=answers_file,
        vocab=full_train_dataset.vocab,
        max_length=max_length,
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (train_loader, val_loader, test_loader, 
            full_train_dataset.vocab_size, full_train_dataset.num_classes,
            full_train_dataset.vocab)


class MultimodalVQADataset(Dataset):
    """
    Multimodal VQA Dataset that loads both images and questions
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        answers_file: str,
        vocab: Optional[Dict] = None,
        answer_to_idx: Optional[Dict] = None,
        max_length: int = 32,
        image_size: int = 224,
        mode: str = 'train'
    ):
        """
        Args:
            csv_file: Path to CSV with question, answer, image columns
            image_dir: Directory containing images
            answers_file: Path to file with all possible answers
            vocab: Word vocabulary (if None, builds from data)
            answer_to_idx: Answer vocabulary (if None, builds from answers_file)
            max_length: Maximum question length
            image_size: Size to resize images
            mode: 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.max_length = max_length
        self.image_size = image_size
        self.mode = mode
        
        # Build answer vocabulary
        if answer_to_idx is None:
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers = [line.strip() for line in f.readlines()]
            # Use the proper answer vocabulary builder
            self.answer_to_idx, self.idx_to_answer = build_answer_vocab(answers)
        else:
            self.answer_to_idx = answer_to_idx
        
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}
        self.num_classes = len(self.answer_to_idx)
        
        # Build question vocabulary
        if vocab is None:
            self.vocab = self._build_vocabulary()
        else:
            self.vocab = vocab
        
        self.vocab_size = len(self.vocab)
        
        # Image transforms
        from torchvision import transforms
        
        if mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        print(f"Loaded {len(self.data)} samples")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Image size: {image_size}x{image_size}")
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from questions"""
        from collections import Counter
        
        word_counter = Counter()
        for question in self.data['question']:
            words = question.lower().split()
            word_counter.update(words)
        
        # Keep most common words
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for word, _ in word_counter.most_common(10000):
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def _encode_question(self, question: str) -> torch.Tensor:
        """Encode question to tensor"""
        words = question.lower().split()
        
        # Convert to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image, question, and answer"""
        row = self.data.iloc[idx]
        
        # Load and transform image
        image_name = row['image']
        if not image_name.endswith('.png'):
            image_name = f"{image_name}.png"
        
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return black image on error
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Encode question
        question = row['question']
        question_tensor = self._encode_question(question)
        
        # Get answer
        answer_text = str(row['answer']).strip()
        answer_idx = self.answer_to_idx.get(answer_text, self.answer_to_idx.get('<UNK>', 0))  # Use <UNK> token if unknown
        
        return {
            'image': image,
            'question': question_tensor,
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'question_text': question,
            'answer_text': answer_text,
            'image_name': image_name
        }


def create_multimodal_dataloaders(
    train_csv: str,
    test_csv: str,
    image_dir: str,
    answers_file: str,
    batch_size: int = 8,
    val_split: float = 0.15,
    num_workers: int = 0,
    max_length: int = 32,
    image_size: int = 224
) -> Tuple:
    """
    Create multimodal dataloaders for training, validation, and testing
    
    Returns:
        (train_loader, val_loader, test_loader, vocab_size, num_classes, vocab, answer_to_idx)
    """
    # Create full training dataset to get vocabulary
    full_train_dataset = MultimodalVQADataset(
        csv_file=train_csv,
        image_dir=image_dir,
        answers_file=answers_file,
        max_length=max_length,
        image_size=image_size,
        mode='train'
    )
    
    # Split into train and validation
    dataset_size = len(full_train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create test dataset (shares vocabulary with train)
    test_dataset = MultimodalVQADataset(
        csv_file=test_csv,
        image_dir=image_dir,
        answers_file=answers_file,
        vocab=full_train_dataset.vocab,
        answer_to_idx=full_train_dataset.answer_to_idx,
        max_length=max_length,
        image_size=image_size,
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (train_loader, val_loader, test_loader,
            full_train_dataset.vocab_size, full_train_dataset.num_classes,
            full_train_dataset.vocab, full_train_dataset.answer_to_idx)
