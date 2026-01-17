# Medical Visual Question Answering (Med-VQA) Project
## WOA7015 Advanced Machine Learning - Alternative Assignment

### Project Overview
This project implements advanced deep learning models for Medical Visual Question Answering using the PathVQA dataset. The system combines computer vision and natural language processing to answer questions about pathology images.

### Dataset
- **Training samples**: 19,755 Q&A pairs from 3,457 pathology images
- **Test samples**: 6,761 Q&A pairs
- **Unique answers**: 4,142 different answers (full vocabulary)
- **Reduced vocabulary**: 1,000 most frequent answers for improved training
- **Question types**: Closed-ended (yes/no) and open-ended questions

### Key Achievements
- **Text Baseline**: 47.36% accuracy (LSTM-based)
- **Original Multimodal**: 41.25% accuracy (ResNet50 + LSTM + concatenation)
- **Improved Multimodal**: 55.39% validation accuracy (enhanced architecture)
- **Performance gain**: 14.14% improvement over original multimodal model

### Project Structure
```
my_projct/
├── data/                          # Dataset files
│   ├── train/                     # Training images (3,457 pathology images)
│   ├── trainrenamed.csv           # Training Q&A pairs (19,755 samples)
│   ├── testrenamed.csv            # Test Q&A pairs (6,761 samples)
│   ├── answers.txt                # Full answer vocabulary (4,142 classes)
│   └── answers_top_1000.txt       # Reduced vocabulary (1,000 classes)
├── src/                           # Source code modules
│   ├── config/                    # Configuration files
│   │   └── improved_training_config.py  # Enhanced training configurations
│   ├── data/                      # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py             # MultimodalVQADataset class
│   │   └── preprocessing.py       # Image preprocessing utilities
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── text_model.py          # LSTM text baseline (47.36% accuracy)
│   │   ├── multimodal_model.py    # Original multimodal (41.25% accuracy)
│   │   ├── improved_multimodal_model.py  # Enhanced multimodal (55.39% accuracy)
│   │   └── vision_encoder.py      # Vision encoders and attention mechanisms
│   ├── training/                  # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py             # Base training class
│   │   ├── multimodal_trainer.py  # Multimodal training with history tracking
│   │   ├── train_text_model.py    # Text baseline training script
│   │   ├── train_improved_model.py  # Enhanced multimodal training
│   │   └── quick_train.py         # Quick training utilities
│   ├── evaluation/                # Evaluation modules
│   │   ├── __init__.py
│   │   └── metrics.py             # VQA evaluation metrics
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logger.py              # Training logging
│       ├── visualization.py       # Result visualization
│       ├── generate_diagrams.py   # Architecture diagrams
│       └── quick_improvements.py  # Performance enhancement utilities
├── notebooks/                     # Interactive notebooks
│   ├── 01_data_exploration.ipynb  # Dataset analysis and statistics
│   ├── 02_text_baseline_training.ipynb    # Text-only baseline (LSTM)
│   ├── 03_multimodal_training.ipynb       # Original multimodal training
│   ├── 03.1_multimodal_training.ipynb     # Enhanced multimodal training
│   ├── 04_model_fine_tuning.ipynb         # Fine-tuning strategies
│   └── improved_multimodal_training.ipynb # Latest optimized training
├── checkpoints/                   # Model checkpoints
│   ├── text_baseline_lstm_notebook/  # Text baseline checkpoints
│   └── multimodal_concat/         # Multimodal model checkpoints
├── results/                       # Training results
│   ├── dataset_statistics.json    # Dataset analysis results
│   ├── text_baseline_results.json # Text baseline performance
│   ├── figures/                   # Performance plots and visualizations
│   └── predictions/               # Model prediction outputs
├── reports/                       # Documentation
│   ├── text_baseline_training_report.md     # Text baseline analysis
│   ├── multimodal_implementation_summary.md # Technical implementation details
│   └── Final_Report_Multimodal_VQA.md      # Comprehensive project report
├── experiments/                   # Experimental configurations
│   ├── baseline/                  # Baseline experiment configs
│   └── vlm/                       # Vision-language model experiments
├── compare_models.py              # Model comparison script
├── inference_tool.py              # Standalone inference tool
├── test_models.py                 # Model validation script
├── test_multimodal_setup.py       # Setup validation
├── test_setup.py                  # Environment setup test
├── requirements.txt               # Python dependencies
├── config.yaml                    # Main project configuration
├── config_lightweight.yaml        # Lightweight training config
└── README.md                      # This documentation
```
├── checkpoints/                   # Model checkpoints
├── results/                       # Results and outputs
│   ├── figures/                   # Plots and figures
│   └── predictions/               # Model predictions
├── reports/                       # Project reports
│   ├── preliminary_report.md      # Week 9 preliminary report
│   └── final_report.md            # Week 13/14 final report
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Setup Instructions

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Data Preparation
Data is already organized in the `data/` folder:
- Training images: `data/train/` (3,457 pathology images)
- Training Q&A pairs: `data/trainrenamed.csv` (19,755 samples)
- Test Q&A pairs: `data/testrenamed.csv` (6,761 samples)
- Full vocabulary: `data/answers.txt` (4,142 classes)
- Reduced vocabulary: `data/answers_top_1000.txt` (1,000 classes - recommended)

#### 3. Quick Start Training

**Option 1: Use the Improved Training Notebook (Recommended)**
```bash
# Run the latest optimized training with all improvements make sure you are using the following paths 
"/content/drive/MyDrive/WOA7015 Advanced Machine Learning"
"/content/drive/MyDrive/WOA7015 Advanced Machine Learning/my_projct"
"/content/drive/MyDrive/WOA7015 Advanced Machine Learning/data" 
"/content/drive/MyDrive/my_projct"
"/content/drive/MyDrive/data"
jupyter notebook notebooks/improved_multimodal_training.ipynb
```

**Option 2: Run Specific Training Scripts**
```bash
# Train enhanced multimodal model with improved architecture
python src/training/train_improved_model.py

# Train text baseline (LSTM model)  
python src/training/train_text_model.py

# Compare different model architectures
python compare_models.py
```

**Option 3: Use the Inference Tool**
```bash
# Run inference on trained models
python inference_tool.py
```

#### 4. Expected Performance
- **Text Baseline (LSTM)**: 47.36% accuracy
- **Original Multimodal**: 41.25% accuracy  
- **Enhanced Multimodal**: **55.39% accuracy** (14.14% improvement)

*Results achieved with reduced vocabulary (1,000 classes) and enhanced architecture including cross-modal attention, spatial attention, and trainable ResNet50.*

### Models Implemented

#### 1. Text Baseline Model (`src/models/text_model.py`)
- **Architecture**: LSTM-based text encoder with MLP classifier
- **Performance**: 47.36% accuracy on 1,000 class vocabulary
- **Features**: Question embedding → LSTM → Classification head
- **Use case**: Baseline for text-only question answering

#### 2. Original Multimodal Model (`src/models/multimodal_model.py`)
- **Architecture**: ResNet50 (frozen) + LSTM + Concatenation fusion
- **Performance**: 41.25% accuracy on 1,000 class vocabulary
- **Features**: Simple early fusion of visual and textual features
- **Limitations**: Frozen visual encoder, basic fusion mechanism

#### 3. Enhanced Multimodal Model (`src/models/improved_multimodal_model.py`) ⭐
- **Architecture**: Trainable ResNet50 + LSTM + Cross-modal attention + Spatial attention
- **Performance**: **55.39% accuracy** on 1,000 class vocabulary (14.14% improvement)
- **Key Features**:
  - Trainable ResNet50 backbone for domain adaptation
  - Cross-modal attention for vision-language interaction
  - Spatial attention for fine-grained image understanding
  - Advanced fusion mechanisms
  - Focal loss for handling class imbalance
- **Parameters**: 33.3M trainable parameters

#### Key Improvements in Enhanced Model:
- **Visual Processing**: Trainable ResNet50 enables pathology-specific feature learning
- **Attention Mechanisms**: Cross-modal and spatial attention improve focus on relevant regions
- **Training Strategy**: Differential learning rates, focal loss, enhanced augmentation
- **Vocabulary Optimization**: Reduced from 4,142 to 1,000 classes for better generalization

### Key Features of the Codebase

#### Configuration
- **Improved Training Configuration**: The `improved_training_config.py` file introduces enhancements such as reduced class complexity, better data augmentation, and optimized learning rates.

#### Dataset
- **PathVQADataset**: Located in `dataset.py`, this class handles loading and preprocessing of the PathVQA dataset, including building answer vocabularies and applying transformations.

#### Training
- **Improved Training Script**: The `train_improved_model.py` script integrates the improved multimodal model with the existing data pipeline, featuring updated configurations for batch size, learning rate, and early stopping.

#### Models
- **Improved Multimodal VQA Model**: The `improved_multimodal_model.py` file defines a model with cross-modal attention fusion, spatial attention for vision, and multi-head cross-attention mechanisms.

#### Evaluation
- **VQA Metrics**: The `metrics.py` file provides evaluation metrics such as accuracy, F1-score, precision, and recall for assessing model performance.

### Evaluation Metrics & Results

#### Primary Metrics
- **Classification Accuracy**: Exact match between predicted and ground truth answers
- **Top-k Accuracy**: Prediction within top-k most likely answers
- **Loss Tracking**: Cross-entropy loss for training convergence monitoring

#### Current Results Summary
| Model | Vocabulary Size | Accuracy | Improvement | Status |
|-------|----------------|----------|-------------|---------|
| Text Baseline (LSTM) | 1,000 classes | 47.36% | - | ✅ Complete |
| Original Multimodal | 1,000 classes | 41.25% | - | ✅ Complete |
| **Enhanced Multimodal** | 1,000 classes | **55.39%** | **+14.14%** | ✅ **Best Performance** |

#### Performance Analysis
- **Dataset**: PathVQA with 3,457 training images, 19,755 Q&A pairs
- **Challenge**: Medical VQA requiring both visual understanding and domain knowledge
- **Key Insight**: Vocabulary reduction (4,142→1,000) significantly improved performance
- **Best Strategy**: Enhanced multimodal with trainable backbone + attention mechanisms

