# Medical Visual Question Answering (Med-VQA) Project
## WOA7015 Advanced Machine Learning - Alternative Assignment

### Project Overview
This project implements deep learning models for Medical Visual Question Answering using the PathVQA dataset. The system answers natural language questions about pathology images.

### Dataset
- **Training samples**: 19,755 Q&A pairs from 3,457 pathology images
- **Test samples**: 6,761 Q&A pairs
- **Unique answers**: 4,593 different answers
- **Question types**: Closed-ended (yes/no) and open-ended questions

### Project Structure
```
my_projct/
├── data/                          # Data files
│   ├── train/                     # Training images
│   ├── trainrenamed.csv           # Training Q&A pairs
│   ├── testrenamed.csv            # Test Q&A pairs
│   └── answers.txt                # Unique answer vocabulary
├── src/                           # Source code
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset classes
│   │   ├── preprocessing.py       # Image preprocessing
│   │   └── augmentation.py        # Data augmentation
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py        # CNN baseline model
│   │   ├── vlm_model.py           # Vision-Language Model
│   │   └── attention.py           # Attention mechanisms
│   ├── training/                  # Training scripts
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   └── config.py              # Training configurations
│   ├── evaluation/                # Evaluation scripts
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── evaluator.py           # Model evaluation
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── logger.py              # Logging utilities
│       └── visualization.py       # Visualization tools
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Data analysis
│   ├── 02_baseline_model.ipynb    # Baseline experiments
│   └── 03_advanced_model.ipynb    # Advanced experiments
├── experiments/                   # Experiment outputs
│   ├── baseline/                  # Baseline results
│   └── vlm/                       # VLM results
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
Data is already organized in the `train/` folder with CSV annotations.

#### 3. Training Models
```bash
# Train baseline CNN model
python src/training/train_baseline.py

# Train VLM model
python src/training/train_vlm.py
```

#### 4. Evaluation
```bash
# Evaluate model
python src/evaluation/evaluate.py --model_path checkpoints/best_model.pth
```

### Models Implemented
1. **Baseline CNN Model**: ResNet + LSTM for question encoding + MLP classifier
2. **Vision-Language Model**: Fine-tuned BLIP/ViLT for Med-VQA

### Evaluation Metrics
- **Closed-ended questions**: Accuracy, F1-score
- **Open-ended questions**: BLEU, ROUGE, Exact Match
- **Overall**: Weighted accuracy

### Timeline
- **Week 9**: Preliminary report with baseline results
- **Week 13/14**: Final report with comprehensive analysis

### Team Members
- [Add team member names and contributions]

### References
- PathVQA Dataset: [Add citation]
- Model architectures: [Add citations]

### Contact
For questions or cloud token requests: Prof. Ir. Dr. Chan Chee Seng - cs.chan@um.edu.my
