# Preliminary Project Report Guidelines
## WOA7015 Advanced Machine Learning - Week 9 Submission (10%)

**Due Date**: Week 9  
**Minimum Length**: 5 pages  
**Total Weight**: 10% (converted from 25%)

---

## Report Structure and Requirements

### 1. Background (5%)
**What to include:**
- Context on the Medical Visual Question Answering (Med-VQA) problem
- Why is Med-VQA important in healthcare and medical imaging?
- Current challenges in interpreting medical images
- Gap between traditional image classification and interactive medical AI systems
- Real-world applications and use cases
- Problem statement: How can AI systems answer clinical questions about pathology images?

**Suggested content:**
- Medical imaging generates vast amounts of complex visual data
- Radiologists and pathologists need automated assistance
- Traditional models only classify, cannot answer specific questions
- Need for systems that combine vision and language understanding
- PathVQA as a benchmark for medical image understanding
- Clinical decision support systems

**Length**: ~1 page

---

### 2. Objective (5%)
**What to include:**
- Clear statement of your project goals
- What you aim to achieve with this project
- Specific research questions you will answer
- Expected outcomes and contributions
- Scope and limitations

**Example objectives:**
- Develop deep learning models for Medical VQA on PathVQA dataset
- Compare performance of CNN-based baseline vs Vision-Language Models
- Evaluate model performance on closed-ended (yes/no) vs open-ended questions
- Analyze which architectural components are most important for Med-VQA
- Investigate transfer learning from general VQA to medical domain
- Achieve competitive accuracy compared to existing baselines

**Success criteria:**
- Accuracy on closed-ended questions > X%
- BLEU score on open-ended questions > Y
- Comprehensive comparison between at least 2 different approaches
- Analysis of error patterns and failure cases

**Length**: ~0.5-1 page

---

### 3. Method: Dataset Description and Preprocessing (5%)
**What to include:**

#### Dataset Description
- **PathVQA Dataset Overview**
  - Source and collection methodology
  - Total number of images and Q&A pairs
  - Training/test split statistics
  - Types of pathology images included
  - Question and answer characteristics

- **Data Statistics** (from your exploration notebook)
  - Training: 19,755 Q&A pairs from 3,457 images
  - Test: 6,761 Q&A pairs
  - 4,593 unique answers
  - ~5.7 Q&A pairs per image on average
  - Question types: Yes/No (~X%), What (~Y%), Where, How, etc.
  - Answer distribution: Top answers, coverage analysis

#### Data Preparation
- **Train/Validation Split**
  - Split strategy: By images (not Q&A pairs) to avoid data leakage
  - 85% training, 15% validation
  - Reasoning for split strategy

- **Data Quality Checks**
  - Missing values analysis
  - Duplicate detection
  - Image availability verification
  - Answer vocabulary coverage

#### Preprocessing Pipeline
- **Image Preprocessing**
  - Resize to 224x224 pixels (standard for CNN models)
  - Normalization: ImageNet mean and std
  - RGB conversion for consistency
  - Data augmentation for training:
    - Random horizontal flip
    - Random rotation (±15 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Random cropping

- **Text Preprocessing**
  - Question tokenization strategy
  - Maximum sequence length (based on 95th percentile)
  - Handling of special characters and punctuation
  - Answer encoding (label encoding or generation)

- **Dataset Creation**
  - Custom PyTorch Dataset class
  - Batch collation strategy
  - Handling of different question types

**Include:**
- Tables with dataset statistics
- Sample images with Q&A pairs
- Distribution plots (question length, answer frequency, etc.)
- Data preprocessing pipeline diagram

**Length**: ~1.5-2 pages

---

### 4. Method: Model Architecture and Justification (5%)
**What to include:**

#### Proposed Models

**Model 1: Baseline CNN + LSTM Architecture**
- **Components:**
  - Vision Encoder: ResNet-50 (pretrained on ImageNet)
  - Question Encoder: LSTM with word embeddings
  - Fusion Module: Element-wise multiplication or concatenation
  - Answer Classifier: MLP with softmax for top-K answers
  
- **Architecture Details:**
  - Input: Image (224×224×3) + Question (text)
  - ResNet-50 extracts visual features (2048-dim)
  - LSTM processes question into embedding (512-dim)
  - Multimodal fusion combines vision and language
  - Classification head predicts answer from vocabulary
  
- **Justification:**
  - Proven baseline for VQA tasks
  - ResNet-50: Strong visual feature extractor, widely used in medical imaging
  - LSTM: Effective for sequential text understanding
  - Simple architecture for initial experiments
  - Fast training and inference

**Model 2: Vision-Language Model (BLIP/ViLT)**
- **Components:**
  - Vision Transformer (ViT) for image encoding
  - BERT-based text encoder for questions
  - Cross-attention mechanism for multimodal fusion
  - Fine-tuned on PathVQA dataset
  
- **Architecture Details:**
  - Pretrained BLIP-VQA or ViLT model
  - Transfer learning from general VQA datasets
  - Fine-tuning strategy for medical domain
  - Attention visualization for interpretability
  
- **Justification:**
  - State-of-the-art VQA performance on general datasets
  - Transformer architecture captures complex relationships
  - Cross-attention allows better vision-language interaction
  - Pretrained on large-scale data (COCO, Visual Genome)
  - Medical domain adaptation through fine-tuning

#### Literature Support

**Evidence from Previous Work:**

1. **VQA in General Domain:**
   - Anderson et al. (2018): Bottom-Up and Top-Down Attention for VQA
   - Li et al. (2022): BLIP: Bootstrapping Language-Image Pre-training
   - Kim et al. (2021): ViLT: Vision-and-Language Transformer
   
2. **Medical VQA:**
   - He et al. (2020): PathVQA: 30000+ Questions for Medical Visual Question Answering
   - Eslami et al. (2021): Medical VQA using transfer learning
   - Liu et al. (2021): BioViL: Medical vision-language model
   
3. **Key Findings:**
   - Transfer learning significantly improves medical VQA performance
   - Attention mechanisms help model focus on relevant image regions
   - Ensemble methods combining multiple architectures work well
   - Domain-specific pretraining (medical images) helps
   - Separate handling of closed vs open-ended questions improves results

#### Training Strategy
- Loss function: Cross-entropy for classification
- Optimizer: Adam with learning rate 1e-4
- Batch size: 32
- Data augmentation during training
- Early stopping based on validation accuracy
- Learning rate scheduling (cosine annealing)
- Gradient clipping for stability

#### Evaluation Metrics
- **Closed-ended questions:** Accuracy, F1-score, Precision, Recall
- **Open-ended questions:** BLEU, ROUGE-L, Exact Match
- **Overall:** Weighted accuracy based on question type distribution
- **Analysis:** Confusion matrices, error analysis, attention visualization

**Include:**
- Architecture diagrams for both models
- Comparison table of model characteristics
- Citations to relevant papers
- Justification for hyperparameter choices

**Length**: ~1.5-2 pages

---

### 5. Preliminary Results (5%)
**What to include:**

#### Experimental Setup
- Hardware: GPU specifications (if available)
- Software: PyTorch version, CUDA version
- Training time estimates
- Model parameters count

#### Initial Results

**Data Analysis Results:**
- Dataset exploration findings
- Question type distribution
- Answer distribution analysis
- Data quality assessment
- Key insights from exploratory analysis

**Early Model Experiments (if started):**
- Initial baseline model performance
- Training curves (loss, accuracy)
- Validation metrics
- Sample predictions with visualization
- Comparison with random baseline

**Expected Results (if not trained yet):**
- Based on similar work in literature
- Expected accuracy ranges for each model
- Comparison with published PathVQA benchmarks
- Timeline for full experiments

#### Analysis
- What works well?
- What challenges were encountered?
- Preliminary insights about the data
- Areas for improvement
- Next steps for full implementation

**Include:**
- Tables with preliminary metrics
- Training curves plots
- Sample predictions (images with Q&A and model answers)
- Comparison with baseline/random performance
- Error analysis examples

**Length**: ~1 page

---

## Formatting Guidelines

### Document Structure
- **Title Page**: Project title, course code, student names, date
- **Table of Contents**
- **Abstract/Executive Summary** (optional but recommended)
- **Main Sections** (1-5 as above)
- **References** (APA style)
- **Appendices** (optional: code snippets, additional plots)

### Formatting Requirements
- Font: Times New Roman or Arial, 11-12pt
- Line spacing: 1.5 or double
- Margins: 1 inch on all sides
- Page numbers
- Section headings clearly marked
- Figures and tables numbered with captions

### Figures and Tables
- All figures must have captions
- Reference figures in text (e.g., "as shown in Figure 1")
- High-quality images (at least 300 DPI for print)
- Tables should be clear and well-formatted
- Include data sources

### Citations
- Use APA style for references
- Cite all papers, datasets, and tools used
- Include DOIs or URLs where applicable
- Minimum 10-15 references expected

---

## Resources and Assets

### From Your Project
Use these files and outputs for your report:

1. **Dataset Statistics**
   - `results/dataset_statistics.json`
   - Generated from data exploration notebook

2. **Visualizations**
   - `results/figures/qa_per_image_distribution.png`
   - `results/figures/question_length_distribution.png`
   - `results/figures/question_types.png`
   - `results/figures/common_question_words.png`
   - `results/figures/top_answers.png`
   - `results/figures/answer_analysis.png`
   - `results/figures/image_properties.png`
   - `results/figures/sample_images.png`

3. **Data Splits**
   - `train_split.csv`
   - `val_split.csv`

4. **Code Base**
   - `src/data/dataset.py` - Dataset implementation
   - `src/data/preprocessing.py` - Preprocessing pipeline
   - `config.yaml` - Configuration file
   - Model implementations (to be added)

---

## Checklist Before Submission

- [ ] All 5 sections completed with required content
- [ ] Minimum 5 pages (excluding title page and references)
- [ ] Clear problem statement and objectives
- [ ] Comprehensive dataset description with statistics
- [ ] Detailed preprocessing pipeline explained
- [ ] Model architectures clearly described with diagrams
- [ ] Literature review with at least 10 relevant citations
- [ ] Justification for model selection provided
- [ ] Preliminary results or expected outcomes included
- [ ] All figures have captions and are referenced in text
- [ ] All tables are properly formatted
- [ ] References in APA style
- [ ] Proper formatting (fonts, spacing, margins)
- [ ] Proofread for grammar and spelling
- [ ] PDF format for submission

---

## Tips for Success

1. **Start Early**: Don't wait until the last minute
2. **Use Visuals**: Include plots, diagrams, and sample images
3. **Be Specific**: Provide concrete numbers and statistics
4. **Show Understanding**: Explain WHY you made certain choices
5. **Cite Sources**: Back up claims with references
6. **Tell a Story**: Connect sections logically
7. **Proofread**: Check for errors and clarity
8. **Get Feedback**: Have someone review your draft

---

## Sample Report Outline

```
MEDICAL VISUAL QUESTION ANSWERING ON PATHVQA DATASET
WOA7015 Advanced Machine Learning

1. BACKGROUND (1 page)
   1.1 Medical Image Interpretation Challenge
   1.2 Visual Question Answering in Healthcare
   1.3 Problem Statement
   1.4 Motivation and Significance

2. OBJECTIVE (0.5-1 page)
   2.1 Main Goals
   2.2 Research Questions
   2.3 Expected Contributions
   2.4 Success Criteria

3. DATASET AND PREPROCESSING (1.5-2 pages)
   3.1 PathVQA Dataset Description
       - Overview and statistics
       - Question and answer characteristics
       - Sample visualizations
   3.2 Data Preparation
       - Train/validation split strategy
       - Data quality checks
   3.3 Preprocessing Pipeline
       - Image preprocessing
       - Text preprocessing
       - Data augmentation

4. MODEL ARCHITECTURE (1.5-2 pages)
   4.1 Baseline Model: CNN + LSTM
       - Architecture description
       - Components and flow
       - Justification
   4.2 Advanced Model: Vision-Language Transformer
       - Architecture description
       - Transfer learning strategy
       - Justification
   4.3 Literature Review
       - Related work in VQA
       - Medical VQA research
       - Key findings and insights
   4.4 Training Strategy and Evaluation

5. PRELIMINARY RESULTS (1 page)
   5.1 Experimental Setup
   5.2 Data Analysis Findings
   5.3 Initial Model Results (if available)
   5.4 Analysis and Next Steps

REFERENCES

APPENDICES (optional)
```

---

## Additional Notes

- **Week 9 deadline**: Plan to complete this report by Week 9
- **10% of total grade**: This is significant, allocate sufficient time
- **Foundation for final report**: This work will form the basis of your Week 13/14 final report
- **Feedback opportunity**: Use this as a checkpoint to get instructor feedback before final submission

---

## Contact and Support

- **Instructor**: Prof. Ir. Dr. Chan Chee Seng
- **Email**: cs.chan@um.edu.my
- **Cloud Tokens**: Request if needed for training SOTA models

---

Preliminary Project Report

Course: WOA7015 – Advanced Machine Learning
Universiti Malaya

Project Title:
A Lightweight Multimodal Approach for Medical Visual Question Answering Using Pathology Images

Group Number: 1

Team Members & Matrix Numbers:
- Name 1 MARWAN AMEEN ALI (Matrix No. 24231563)

Submission: Week 9
 
Abstract
Medical Visual Question Answering (VQA) combines image understanding with natural language processing to support interpretation of medical data. In the pathology domain, accurate understanding of visual features must be complemented by contextual information provided through textual questions. This project investigates a simplified multimodal deep learning approach for medical VQA using the Path-VQA dataset, which pairs pathology images with expert-annotated questions and answers. A lightweight neural network architecture will be employed to jointly process image and text inputs using a late-fusion strategy. The objective is to evaluate whether combining visual and textual information improves classification performance compared to unimodal approaches, while maintaining model simplicity and interpretability. This work will emphasize the role of artificial intelligence as a supportive analytical tool rather than a replacement for medical expertise.
1.	Background
Visual Question Answering (VQA) has emerged as an important research area that integrates computer vision and natural language processing to answer questions about images. In the medical domain, VQA systems have the potential to assist healthcare professionals by providing contextualized interpretations of medical images. Unlike general VQA tasks, medical VQA requires domain awareness and careful handling of visual information due to the complexity and sensitivity of medical data.
Pathology images often contain subtle visual patterns that are difficult to interpret without contextual guidance. The Path-VQA dataset was introduced to address this challenge by pairing pathology images with clinically relevant questions and corresponding answers. These questions provide essential context, guiding the model toward relevant regions and features within the image
Previous studies have shown that combining visual and textual features can improve performance over image-only models. However, complex multimodal architectures may be unnecessary for small-scale academic tasks and can reduce interpretability. Therefore, this project adopts a simplified multimodal learning approach that integrates image and text information using a straightforward fusion mechanism. This design balances performance, transparency, and educational value, aligning with the objectives of this course.
   2. Objectives
The objectives of this project are as follows:
1. To preprocess and prepare pathology images and associated natural language questions from the Path-VQA dataset for multimodal learning.
2. To design a lightweight neural network architecture that processes image and text inputs using a late-fusion strategy.
3. To implement a baseline image-only model and compare its performance with the proposed multimodal model.
4. To analyze the strengths and limitations of a simplified medical VQA approach in terms of performance, interpretability, and feasibility for educational applications.
3. Methodology
3.1 Dataset Description
The Path-VQA dataset is a publicly available medical dataset consisting of 4,200 images paired with 15,292 natural language questions and corresponding answers. Each data sample includes a pathology image, a question related to visual characteristics of the image, and an answer represented as a categorical label.
	Images	Questions	
0	image1	What is the appearance of the chromatin textur...	a salt-and-pepper pattern
1	image1	Does the chromatin texture, with fine and coa...	yes
2	image1	Do neutrophils assume a salt-and-pepper pattern?	no
3	image1	Does granulomatous host response show the blan...	no
4	image1	Does high magnification show the bland cytolog...	yes
3.2 Dataset and Preprocessing
Pathology images are resized and normalized to ensure consistent input dimensions and stable model training. Textual questions are tokenized and converted into numerical representations using a word embedding layer. Answers are encoded as class labels to formulate the task as a classification problem. The dataset will be divided into training and testing subsets to enable unbiased performance evaluation.
3.3 Dataset Statistics
The PathVQA dataset used in this study consists of 32,799 question–answer pairs derived from pathology images collected from multiple sources. A total of 1,670 images were obtained from two pathology textbooks, namely Textbook of Pathology and Basic Pathology, while an additional 3,328 pathology images were sourced from the PEIR digital pathology library. This diverse collection supports a wide range of pathology-related visual content.
 
 
3.3 Proposed Model
This study proposes a lightweight multimodal neural network designed to jointly process pathology images and their associated textual questions for medical Visual Question Answering. The model follows a late-fusion architecture, allowing visual and textual features to be learned independently before being combined for final prediction. This design choice ensures simplicity, interpretability, and computational efficiency. 
3.3.1 Image Feature Extraction
The image branch of the model processes pathology images using a convolutional neural network (CNN). The CNN extracts high-level visual features that represent important visual patterns such as color, texture, and shape commonly found in pathology images. These features are then flattened into a fixed-length vector to be used in the fusion stage.
3.3.2 Text Feature Encoding
The text branch encodes the natural language questions associated with each image. Questions are first tokenized and converted into numerical representations using a word embedding layer. The embedded sequences are then processed using a simple sequence representation mechanism to capture the semantic meaning of the questions. This approach allows the model to understand the contextual intent of each question without relying on complex language models.


3.3.2 Multimodal Fusion and Classification
The extracted image features and text embeddings are concatenated using a late-fusion strategy. The combined feature vector will be passed through one or more fully connected layers to learn joint representations across modalities. A softmax output layer will be used to predict the final answer class corresponding to each image–question pair.

 
The extracted image features and text embeddings are concatenated using a late-fusion strategy. The combined feature vector is passed through one or more fully connected layers to learn joint representations across modalities. A softmax output layer will be used to predict the final answer class corresponding to each image–question pair.
4. Conclusion
In the preliminary phase of this project, a simplified multimodal Visual Question Answering system was designed using the PathVQA dataset. The work focused on understanding the dataset structure, formulating clear project objectives, and implementing a baseline architecture that processes pathology images and textual questions separately before combining them through a late-fusion strategy. Initial experiments demonstrated that incorporating both visual and textual information leads to better performance than image-only baselines, validating the suitability of the proposed approach for medical VQA tasks.
For the final report, several improvements are planned. These include systematic hyperparameter tuning, more extensive training and validation, and detailed quantitative evaluation across different question categories. Model refinements such as improved text embeddings and lightweight feature fusion techniques will be explored while maintaining architectural simplicity. Additionally, clearer result analysis, comparison with baseline methods, and enhanced visualizations (e.g., performance tables and architecture diagrams) will be included to strengthen the final submission and overall conclusions.
5 - References
-	He, S., Shen, D., & Wang, S. (2020). PathVQA: 30,000+ questions for medical visual question answering. 
