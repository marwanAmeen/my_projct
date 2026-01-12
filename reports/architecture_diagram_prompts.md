# Architecture Diagram Prompts for Med-VQA Models

## Model 1: Baseline CNN + LSTM Architecture

### Prompt for AI Image Generation Tools (DALL-E, Midjourney):
```
Create a technical architecture diagram for a Medical Visual Question Answering system with the following components:
- Left side: Input pathology medical image (224x224x3) flowing into ResNet-50 CNN
- ResNet-50 outputs visual features (2048-dimensional vector)
- Right side: Input question text flowing into Word Embedding layer, then LSTM encoder
- LSTM outputs question features (512-dimensional vector)
- Both visual and question features merge in a Fusion Module (element-wise multiplication)
- Fused features (512-dim) flow into MLP classifier with multiple layers
- Output: Answer prediction (softmax over 4593 classes)
- Use clean boxes and arrows, professional blue and gray color scheme
- Show dimensions at each layer
- Include labels: "Image Encoder", "Question Encoder", "Multimodal Fusion", "Answer Classifier"
- Style: Clean technical diagram, machine learning architecture visualization
```

### Detailed Text Description for Manual Drawing:
```
BASELINE MODEL ARCHITECTURE (CNN + LSTM)

INPUT LAYER:
├─ Medical Image: [224 × 224 × 3]
└─ Question Text: "What is shown in the image?"

IMAGE PROCESSING BRANCH:
Medical Image [224×224×3]
    ↓
ResNet-50 (pretrained, frozen/fine-tuned)
    ↓
Global Average Pooling
    ↓
Visual Features [2048-dim]
    ↓
Linear Projection [2048 → 512]
    ↓
ReLU + Dropout (0.3)
    ↓
Visual Embedding [512-dim]

QUESTION PROCESSING BRANCH:
Question Text (tokenized)
    ↓
Word Embedding [vocab_size × 300]
    ↓
LSTM Encoder (2 layers, hidden_dim=256)
    ↓
Final Hidden State
    ↓
Linear Projection [256 → 512]
    ↓
ReLU + Dropout (0.3)
    ↓
Question Embedding [512-dim]

MULTIMODAL FUSION:
Visual Embedding [512-dim] ─┐
                            ├─→ Element-wise Multiplication
Question Embedding [512-dim]─┘
    ↓
Fused Features [512-dim]
    ↓
Layer Normalization

ANSWER CLASSIFIER:
Fused Features [512-dim]
    ↓
Linear Layer [512 → 1024]
    ↓
ReLU + Dropout (0.5)
    ↓
Linear Layer [1024 → 512]
    ↓
ReLU + Dropout (0.5)
    ↓
Output Layer [512 → 4593]
    ↓
Softmax
    ↓
Predicted Answer

DIMENSIONS SUMMARY:
- Input Image: 224 × 224 × 3
- ResNet-50 features: 2048-dim
- LSTM hidden: 256-dim
- Embedding space: 512-dim
- Number of classes: 4593
- Total parameters: ~28M
```

---

## Model 2: Vision-Language Transformer (BLIP/ViLT)

### Prompt for AI Image Generation Tools:
```
Create a technical architecture diagram for a Vision-Language Transformer model for Medical VQA:
- Left side: Input medical image (224x224) going into Vision Transformer (ViT) with patch embeddings
- Show image being split into 16x16 patches
- ViT processes patches with multi-head self-attention blocks (12 layers)
- Right side: Input question text going into BERT tokenizer then BERT encoder
- BERT processes text with transformer blocks (12 layers)
- Middle: Cross-attention fusion module connecting vision and language features
- Show bidirectional arrows between vision and text representations
- Bottom: MLP head for answer prediction
- Use modern transformer architecture style with attention mechanisms visualized
- Color scheme: Teal for vision, purple for text, orange for fusion
- Include layer counts and dimension annotations
- Style: Modern ML architecture, similar to attention is all you need paper diagrams
```

### Detailed Text Description for Manual Drawing:
```
VISION-LANGUAGE TRANSFORMER (BLIP-VQA)

INPUT LAYER:
├─ Medical Image: [224 × 224 × 3]
└─ Question Text: "Is there evidence of malignancy?"

IMAGE ENCODER (Vision Transformer):
Medical Image [224×224×3]
    ↓
Patch Embedding (16×16 patches)
    ↓
Patch Tokens [196 × 768] + [CLS] token
    ↓
Position Encoding Added
    ↓
┌─────────────────────────┐
│ ViT Encoder (12 layers) │
│  ┌──────────────────┐   │
│  │ Multi-Head       │   │
│  │ Self-Attention   │   │
│  │ (12 heads)       │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │ Layer Norm       │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │ Feed Forward     │   │
│  │ (MLP)            │   │
│  └──────────────────┘   │
│         ↓               │
│  └── Repeat 12x ────┘   │
└─────────────────────────┘
    ↓
Visual Features [197 × 768]
(196 patches + 1 CLS token)

TEXT ENCODER (BERT):
Question Text (tokenized)
    ↓
Token Embedding [vocab_size × 768]
    ↓
Position Encoding Added
    ↓
Segment Embedding Added
    ↓
┌─────────────────────────┐
│ BERT Encoder (12 layers)│
│  ┌──────────────────┐   │
│  │ Multi-Head       │   │
│  │ Self-Attention   │   │
│  │ (12 heads)       │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │ Layer Norm       │   │
│  └──────────────────┘   │
│         ↓               │
│  ┌──────────────────┐   │
│  │ Feed Forward     │   │
│  │ (MLP)            │   │
│  └──────────────────┘   │
│         ↓               │
│  └── Repeat 12x ────┘   │
└─────────────────────────┘
    ↓
Text Features [seq_len × 768]

CROSS-MODAL FUSION:
┌────────────────────────────────────┐
│    Cross-Attention Mechanism       │
│                                    │
│  Visual Features [197 × 768] ──┐  │
│                                 │  │
│  ┌─────────────────────────┐   │  │
│  │ Query from Text         │←──┘  │
│  │ Key/Value from Vision   │      │
│  └─────────────────────────┘      │
│            ↓                       │
│  ┌─────────────────────────┐      │
│  │ Multi-Head Cross-       │      │
│  │ Attention (6 layers)    │      │
│  └─────────────────────────┘      │
│            ↓                       │
│  Text Features [seq_len × 768]←──┐│
│                                   ││
│  ┌─────────────────────────┐     ││
│  │ Query from Vision       │←────┘│
│  │ Key/Value from Text     │      │
│  └─────────────────────────┘      │
│            ↓                       │
│  Fused Representation [768-dim]   │
└────────────────────────────────────┘

ANSWER PREDICTION HEAD:
[CLS] Token Representation [768-dim]
    ↓
Linear Layer [768 → 512]
    ↓
GELU Activation
    ↓
Layer Norm
    ↓
Dropout (0.1)
    ↓
Linear Layer [512 → 4593]
    ↓
Softmax
    ↓
Predicted Answer

ATTENTION MECHANISMS:
┌──────────────────────────────┐
│ Self-Attention (within mode) │
│ Q, K, V from same sequence   │
└──────────────────────────────┘
┌──────────────────────────────┐
│ Cross-Attention (between)    │
│ Q from text, K,V from vision │
│ Q from vision, K,V from text │
└──────────────────────────────┘

DIMENSIONS SUMMARY:
- Input Image: 224 × 224 × 3
- Patch Size: 16 × 16
- Number of Patches: 196
- Hidden Dimension: 768
- Number of Attention Heads: 12
- ViT Layers: 12
- BERT Layers: 12
- Cross-Attention Layers: 6
- Number of Output Classes: 4593
- Total Parameters: ~340M (BLIP-base)
```

---

## Data Flow Diagram

### Prompt for AI Tools:
```
Create a flowchart showing the complete Med-VQA pipeline:
- Input: Pathology image + Clinical question
- Step 1: Image preprocessing (resize, normalize, augmentation)
- Step 2: Text preprocessing (tokenization, encoding)
- Step 3: Feature extraction (parallel paths for image and text)
- Step 4: Multimodal fusion
- Step 5: Answer generation
- Step 6: Post-processing and output
- Show data flowing from top to bottom
- Include example data at each stage
- Use flowchart style with rectangles for processes, diamonds for decisions
- Clean professional style with consistent colors
```

### Text Description:
```
COMPLETE MED-VQA PIPELINE

START
  ↓
┌─────────────────────────────────┐
│ INPUT DATA                      │
│ - Pathology Image (PNG)         │
│ - Question: "What is shown?"    │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ IMAGE PREPROCESSING             │
│ 1. Load image (PIL)             │
│ 2. Resize to 224×224            │
│ 3. Convert to RGB               │
│ 4. Apply augmentation (train)   │
│    - Random flip                │
│    - Random rotation ±15°       │
│    - Color jitter               │
│ 5. Normalize (ImageNet stats)   │
│ 6. Convert to Tensor            │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ TEXT PREPROCESSING              │
│ 1. Lowercase text               │
│ 2. Tokenization                 │
│ 3. Add special tokens           │
│    [CLS] question [SEP]         │
│ 4. Padding to max_length        │
│ 5. Convert to input_ids         │
│ 6. Create attention_mask        │
└─────────────────────────────────┘
  ↓
┌──────────────┐     ┌──────────────┐
│ IMAGE        │     │ TEXT         │
│ ENCODER      │     │ ENCODER      │
│ (ResNet/ViT) │     │ (LSTM/BERT)  │
└──────────────┘     └──────────────┘
       ↓                    ↓
Visual Features      Text Features
  [2048-dim]           [512-dim]
       └──────────┬──────────┘
                  ↓
       ┌────────────────────┐
       │ MULTIMODAL FUSION  │
       │ - Concatenation    │
       │ - Multiplication   │
       │ - Cross-Attention  │
       └────────────────────┘
                  ↓
          Fused Features
            [512-dim]
                  ↓
       ┌────────────────────┐
       │ ANSWER CLASSIFIER  │
       │ - MLP Layers       │
       │ - Softmax          │
       └────────────────────┘
                  ↓
          Answer Logits
           [4593-dim]
                  ↓
       ┌────────────────────┐
       │ POST-PROCESSING    │
       │ - Argmax           │
       │ - Decode to text   │
       │ - Confidence score │
       └────────────────────┘
                  ↓
┌─────────────────────────────────┐
│ OUTPUT                          │
│ Predicted Answer: "carcinoma"  │
│ Confidence: 0.87                │
└─────────────────────────────────┘
  ↓
END
```

---

## Training Pipeline Diagram

```
TRAINING WORKFLOW

┌──────────────────────────────────────────┐
│ DATA LOADING                             │
│ PathVQADataset                           │
│ - Train: 16,792 samples (85%)           │
│ - Val: 2,963 samples (15%)              │
│ Batch Size: 32                           │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ FORWARD PASS                             │
│ 1. Load batch (images, questions)        │
│ 2. Forward through model                 │
│ 3. Get predictions                       │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ LOSS COMPUTATION                         │
│ - Cross-Entropy Loss                     │
│ - Compare predictions vs ground truth    │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ BACKWARD PASS                            │
│ 1. Compute gradients                     │
│ 2. Clip gradients (max_norm=1.0)        │
│ 3. Update parameters (Adam optimizer)    │
└──────────────────────────────────────────┘
              ↓
        After Each Epoch
              ↓
┌──────────────────────────────────────────┐
│ VALIDATION                               │
│ 1. Evaluate on validation set            │
│ 2. Compute metrics (Acc, F1, BLEU)      │
│ 3. Check early stopping criteria         │
│ 4. Save best model checkpoint            │
└──────────────────────────────────────────┘
              ↓
      ┌──────┴──────┐
      │ Continue?   │
      └──────┬──────┘
         No  │  Yes
             ↓   ↓
          DONE  └──→ Next Epoch
```

---

## Attention Visualization Concept

### Prompt:
```
Create a visualization showing attention mechanisms in Medical VQA:
- Top: Medical pathology image with colorful heatmap overlay
- Heatmap shows which image regions the model focuses on
- Bottom: Question text with highlighted important words
- Middle: Connecting lines showing relationships between text words and image regions
- Example question: "Is there evidence of necrosis?"
- Attention should highlight tissue regions relevant to "necrosis"
- Use warm colors (red, orange) for high attention, cool colors (blue) for low attention
- Professional medical visualization style
```

---

## Comparison Table Visualization

```
MODEL COMPARISON TABLE

┌─────────────────────┬──────────────────┬─────────────────────┐
│ Characteristic      │ Baseline CNN     │ Vision-Language     │
│                     │ + LSTM           │ Transformer (BLIP)  │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Image Encoder       │ ResNet-50        │ Vision Transformer  │
│                     │ (CNN)            │ (ViT)               │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Text Encoder        │ LSTM             │ BERT                │
│                     │ (RNN)            │ (Transformer)       │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Fusion Method       │ Element-wise     │ Cross-Attention     │
│                     │ Multiplication   │ (Bidirectional)     │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Parameters          │ ~28M             │ ~340M               │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Pretrained on       │ ImageNet         │ Large-scale VQA     │
│                     │                  │ (COCO, VG)          │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Training Time       │ ~2-3 hours       │ ~8-10 hours         │
│ (Single GPU)        │                  │                     │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Inference Speed     │ Fast (~50ms)     │ Moderate (~200ms)   │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Memory Requirement  │ ~4GB VRAM        │ ~12GB VRAM          │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Interpretability    │ Limited          │ Good (attention)    │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Expected Accuracy   │ 65-75%           │ 75-85%              │
│ (Yes/No questions)  │                  │                     │
├─────────────────────┼──────────────────┼─────────────────────┤
│ Expected BLEU       │ 30-40            │ 45-55               │
│ (Open-ended)        │                  │                     │
└─────────────────────┴──────────────────┴─────────────────────┘
```

---

## Instructions for Creating Diagrams:

### Using Draw.io / Diagrams.net:
1. Go to https://app.diagrams.net/
2. Create new diagram
3. Use shapes:
   - Rectangles for processing layers
   - Rounded rectangles for input/output
   - Arrows for data flow
   - Dashed boxes for grouping
4. Add text labels with dimensions
5. Export as PNG or SVG

### Using Microsoft PowerPoint:
1. Use SmartArt for flow diagrams
2. Insert shapes for architecture blocks
3. Use connectors for arrows
4. Add text boxes for annotations
5. Group related elements
6. Export as high-resolution image

### Using Python (Matplotlib/Graphviz):
1. See the Python code file I'll create next
2. Automatically generate diagrams from code
3. Easily update and regenerate

### Using LaTeX (TikZ):
1. Use the LaTeX diagram code I'll provide
2. Compile with pdflatex
3. Professional publication-quality output

Would you like me to create Python scripts or LaTeX TikZ code to generate these diagrams programmatically?
```

