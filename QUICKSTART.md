# Quick Start Guide - Text-Only VQA Training

##   What We've Built

### 1. **Dataset Module** ([src/data/dataset.py](src/data/dataset.py))
- `TextOnlyVQADataset` - Loads questions and answers (no images)
- Builds vocabulary from questions
- Encodes text into token IDs
- `create_text_dataloaders()` - Creates train/val/test loaders

### 2. **Model Module** ([src/models/text_model.py](src/models/text_model.py))
- `LSTMTextModel` - Bidirectional LSTM for question encoding
- `TransformerTextModel` - Transformer-based alternative
- Both models predict answer classes from questions

### 3. **Training Module** ([src/training/trainer.py](src/training/trainer.py))
- `TextVQATrainer` - Complete training loop
- Handles training, validation, checkpointing
- Early stopping, learning rate scheduling
- Logs metrics (accuracy, F1, precision, recall)

### 4. **Evaluation Module** ([src/evaluation/metrics.py](src/evaluation/metrics.py))
- `VQAMetrics` - Accuracy, F1, precision, recall
- Per-question-type analysis (yes/no vs open-ended)
- Top-k accuracy, confusion statistics

### 5. **Training Scripts**
- `train_text_model.py` - Main training script
- `quick_train.py` - Fast test (2 epochs)
- `test_setup.py` - Verify setup works

### 6. **Configurations**
- `config.yaml` - Full configuration
- `config_lightweight.yaml` - Optimized for low-spec laptops

---

##   Running with WSL Python

### Step 1: Install Dependencies (in WSL terminal)

```bash
# Navigate to project directory
cd "/mnt/c/Users/AMEE005/Arvato/workspace/um-project/WOA7015 Advanced Machine Learning/my_projct"

# Install required packages (CPU version for low-spec laptop)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install tqdm pyyaml scikit-learn pandas
```

### Step 2: Test the Setup

```bash
# Quick test to verify everything works
python3 test_setup.py
```

This will:
-   Load data and build vocabulary
-   Create LSTM model
-   Test forward pass
-   Test training step

### Step 3: Quick Training Test (2 epochs, ~5-10 minutes)

```bash
python3 quick_train.py
```

This runs a fast training test with:
- Small model (256 hidden dim, 1 layer)
- Small batches (8 samples)
- Only 2 epochs
- CPU optimized

### Step 4: Full Training

```bash
# Option 1: Lightweight LSTM (recommended for low-spec)
python3 train_text_model.py --config config_lightweight.yaml --model-type lstm

# Option 2: Regular LSTM (more epochs, bigger model)
python3 train_text_model.py --model-type lstm

# Option 3: Transformer model (slower but potentially more accurate)
python3 train_text_model.py --config config_lightweight.yaml --model-type transformer
```

### Step 5: Evaluate on Test Set

```bash
python3 train_text_model.py --config config_lightweight.yaml --model-type lstm --eval-test
```

---

##   What to Expect

### Model Size
- **Lightweight LSTM**: ~2-5 MB, ~500K-2M parameters
- **Regular LSTM**: ~10-20 MB, ~5-10M parameters
- **Transformer**: ~20-50 MB, ~10-20M parameters

### Training Time (CPU, low-spec laptop)
- **Quick test (2 epochs)**: 5-10 minutes
- **Lightweight (10 epochs)**: 30-60 minutes
- **Full training (50 epochs)**: 2-4 hours

### Expected Performance
- **Random baseline**: ~0.02% accuracy (1/4593 classes)
- **Text-only LSTM**: 15-30% accuracy
- **Text-only Transformer**: 20-35% accuracy
- *Note: These are WITHOUT images - adding vision will improve significantly!*

---

##   Output Files

### Checkpoints
```
checkpoints/
└── text_baseline_lstm/
    ├── best_model.pth              # Best model based on validation accuracy
    ├── checkpoint_epoch_1.pth      # Checkpoint after each epoch
    ├── checkpoint_epoch_2.pth
    └── training.log                # Training logs
```

### How to Load Trained Model

```python
import torch
from src.models.text_model import create_text_model

# Load checkpoint
checkpoint = torch.load('checkpoints/text_baseline_lstm/best_model.pth')

# Create model
model = create_text_model(
    model_type='lstm',
    vocab_size=checkpoint['config']['text']['vocab_size'],
    num_classes=4593,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=1,
    dropout=0.2,
    bidirectional=True
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
predictions, probabilities = model.predict(questions)
```

---

##   Next Steps (After Text Model Works)

1.   **Text-only baseline** - Training language model (DONE!)
2.   **Add vision**: Implement CNN image encoder
3.   **Multimodal fusion**: Combine text + image features
4.   **Vision-Language Models**: BLIP, CLIP, etc.
5.   **Fine-tuning**: Optimize on PathVQA dataset

---

##   Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "CUDA out of memory"
Use the lightweight config which is CPU-optimized:
```bash
python3 train_text_model.py --config config_lightweight.yaml --model-type lstm
```

### Training is too slow
Reduce batch size and epochs in `config_lightweight.yaml`:
```yaml
training:
  batch_size: 4  # Reduce from 8
  num_epochs: 5  # Reduce from 10
```

### Want to see training progress
The progress bars show:
- Current epoch
- Batch progress
- Loss and accuracy per batch
- Time remaining

---

##   Tips for Low-Spec Laptop

1. **Use lightweight config**: `--config config_lightweight.yaml`
2. **Start with quick test**: Run `quick_train.py` first
3. **Close other programs**: Free up RAM
4. **Use LSTM over Transformer**: LSTM is faster and lighter
5. **Reduce batch size**: Edit config if still slow
6. **Train overnight**: Full training takes time on CPU

---

##   Summary

You now have a complete text-only VQA training pipeline:
-   Data loading and preprocessing
-   LSTM and Transformer models
-   Training loop with validation
-   Metrics and evaluation
-   Checkpointing and logging
-   Low-spec optimizations

**Ready to run!** Start with `test_setup.py` to verify everything works.
