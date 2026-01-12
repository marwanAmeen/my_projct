"""
Generate architecture diagrams for Med-VQA models using Python
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_baseline_architecture_diagram():
    """Create baseline CNN + LSTM architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    input_color = '#E8F4F8'
    vision_color = '#A8D8EA'
    text_color = '#FFCCE5'
    fusion_color = '#FFE5B4'
    output_color = '#D4F4DD'
    
    # Helper function to draw boxes
    def draw_box(x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', 
                            facecolor=color, 
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=fontsize, weight='bold')
    
    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               color='black', linewidth=2)
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 9.5, 'Baseline CNN + LSTM Architecture for Medical VQA',
           ha='center', fontsize=16, weight='bold')
    
    # Input layer
    draw_box(0.5, 8, 2, 0.6, 'Input Image\n224×224×3', input_color)
    draw_box(7.5, 8, 2, 0.6, 'Input Question\n"What is shown?"', input_color)
    
    # Vision branch
    draw_arrow(1.5, 8, 1.5, 7.5)
    draw_box(0.5, 6.5, 2, 0.8, 'ResNet-50\n(Pretrained)', vision_color)
    draw_arrow(1.5, 6.5, 1.5, 6)
    draw_box(0.5, 5.2, 2, 0.6, 'Global Pooling\n2048-dim', vision_color)
    draw_arrow(1.5, 5.2, 1.5, 4.7)
    draw_box(0.5, 4, 2, 0.5, 'Linear + ReLU\n512-dim', vision_color)
    draw_arrow(1.5, 4, 1.5, 3.5)
    draw_box(0.5, 2.8, 2, 0.5, 'Visual Embedding\n512-dim', vision_color)
    
    # Text branch
    draw_arrow(8.5, 8, 8.5, 7.5)
    draw_box(7.5, 6.5, 2, 0.8, 'Word Embedding\nvocab×300', text_color)
    draw_arrow(8.5, 6.5, 8.5, 6)
    draw_box(7.5, 5.2, 2, 0.6, 'LSTM Encoder\n2 layers, h=256', text_color)
    draw_arrow(8.5, 5.2, 8.5, 4.7)
    draw_box(7.5, 4, 2, 0.5, 'Linear + ReLU\n512-dim', text_color)
    draw_arrow(8.5, 4, 8.5, 3.5)
    draw_box(7.5, 2.8, 2, 0.5, 'Question Embedding\n512-dim', text_color)
    
    # Fusion
    draw_arrow(2.5, 3, 4, 2.3)
    draw_arrow(7.5, 3, 6, 2.3)
    draw_box(4, 1.8, 2, 0.8, 'Multimodal Fusion\nElement-wise ×\n512-dim', fusion_color, fontsize=9)
    
    # Classifier
    draw_arrow(5, 1.8, 5, 1.3)
    draw_box(4, 0.5, 2, 0.6, 'MLP Classifier\n512→1024→512', output_color)
    draw_arrow(5, 0.5, 5, 0)
    draw_box(3.5, -0.5, 3, 0.4, 'Output: Answer (4593 classes)', output_color)
    
    # Add legend
    ax.text(0.5, 0.5, 'Vision Path', color=vision_color, 
           bbox=dict(boxstyle='round', facecolor=vision_color, edgecolor='black'))
    ax.text(0.5, 0.2, 'Text Path', color=text_color,
           bbox=dict(boxstyle='round', facecolor=text_color, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig('../results/figures/baseline_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_architecture.png")
    plt.close()


def create_transformer_architecture_diagram():
    """Create Vision-Language Transformer architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    vision_color = '#B3E5FC'
    text_color = '#F8BBD0'
    attention_color = '#FFF9C4'
    output_color = '#C8E6C9'
    
    def draw_box(x, y, width, height, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', 
                            facecolor=color, 
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=fontsize, weight='bold',
               wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, style='->', color='black'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style, mutation_scale=20,
                               color=color, linewidth=2)
        ax.add_patch(arrow)
    
    # Title
    ax.text(6, 11.5, 'Vision-Language Transformer (BLIP-VQA) Architecture',
           ha='center', fontsize=16, weight='bold')
    
    # Input
    draw_box(1, 10, 2, 0.5, 'Image\n224×224×3', '#E0E0E0')
    draw_box(9, 10, 2, 0.5, 'Question\nText', '#E0E0E0')
    
    # Vision branch
    draw_arrow(2, 10, 2, 9.5)
    draw_box(1, 8.8, 2, 0.5, 'Patch Embed\n16×16 patches', vision_color)
    draw_arrow(2, 8.8, 2, 8.3)
    draw_box(0.8, 7, 2.4, 1, 'Vision Transformer\n12 layers\nMulti-Head Attn\nd=768, h=12', vision_color, fontsize=8)
    draw_arrow(2, 7, 2, 6.5)
    draw_box(1, 5.8, 2, 0.5, 'Visual Features\n197×768', vision_color)
    
    # Text branch
    draw_arrow(10, 10, 10, 9.5)
    draw_box(9, 8.8, 2, 0.5, 'Token Embed\n+ Position', text_color)
    draw_arrow(10, 8.8, 10, 8.3)
    draw_box(8.8, 7, 2.4, 1, 'BERT Encoder\n12 layers\nMulti-Head Attn\nd=768, h=12', text_color, fontsize=8)
    draw_arrow(10, 7, 10, 6.5)
    draw_box(9, 5.8, 2, 0.5, 'Text Features\nL×768', text_color)
    
    # Cross-attention
    draw_arrow(3.2, 6, 5.5, 5)
    draw_arrow(8.8, 6, 6.5, 5)
    draw_box(4.5, 4, 3, 1.5, 'Cross-Modal Fusion\nCross-Attention\n6 layers\nVision ⟷ Text', attention_color, fontsize=9)
    
    # Attention detail (small inset)
    draw_box(4.7, 3.2, 2.6, 0.6, 'Q from text, K,V from vision\nQ from vision, K,V from text', '#FFFDE7', fontsize=7)
    
    # Output head
    draw_arrow(6, 4, 6, 3.5)
    draw_box(5, 2.7, 2, 0.6, '[CLS] Token\n768-dim', output_color)
    draw_arrow(6, 2.7, 6, 2.2)
    draw_box(5, 1.5, 2, 0.5, 'MLP Head\n768→512', output_color)
    draw_arrow(6, 1.5, 6, 1)
    draw_box(4.5, 0.3, 3, 0.5, 'Answer Prediction\n4593 classes', output_color)
    
    # Add annotations
    ax.text(2, 5.3, 'Self-Attention', fontsize=8, style='italic')
    ax.text(10, 5.3, 'Self-Attention', fontsize=8, style='italic')
    ax.text(6, 5.7, 'Cross-Modal\nInteraction', fontsize=8, 
           ha='center', style='italic', weight='bold')
    
    # Model info box
    info_text = 'Parameters: ~340M\nHidden Dim: 768\nAttention Heads: 12\nLayers: ViT(12) + BERT(12) + Cross(6)'
    draw_box(0.5, 0.3, 3, 1, info_text, '#F5F5F5', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('../results/figures/transformer_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: transformer_architecture.png")
    plt.close()


def create_training_pipeline_diagram():
    """Create training pipeline flowchart"""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors for different stages
    data_color = '#E1F5FE'
    process_color = '#FFF9C4'
    decision_color = '#FFCCBC'
    save_color = '#C8E6C9'
    
    def draw_box(x, y, width, height, text, color, shape='rect'):
        if shape == 'rect':
            box = FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', 
                                facecolor=color, 
                                linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=9, wrap=True)
    
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               color='black', linewidth=1.5)
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 13.5, 'Medical VQA Training Pipeline', 
           ha='center', fontsize=14, weight='bold')
    
    y = 12.5
    # Data loading
    draw_box(3, y, 4, 0.6, 'Load PathVQA Dataset\nTrain/Val Split (85/15)', data_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Data Preprocessing\nResize, Normalize, Augment', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Initialize Model\n(Baseline or Transformer)', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Epoch Loop Start', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Forward Pass\nGet Predictions', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Compute Loss\nCross-Entropy', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Backward Pass\nGradient Descent', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Validation\nCompute Metrics', process_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    # Decision diamond
    draw_box(3.5, y, 3, 0.6, 'Improved?', decision_color)
    draw_arrow(5, y, 5, y-0.5)
    draw_arrow(6.5, y+0.3, 7.5, y+0.3)
    ax.text(7.8, y+0.3, 'Yes', fontsize=8)
    
    y -= 1.2
    draw_box(7, y, 2.5, 0.5, 'Save Best Model', save_color, fontsize=8)
    
    # Continue or stop
    draw_box(3.5, y, 3, 0.6, 'Max Epochs?', decision_color)
    draw_arrow(3.5, y+0.3, 2, y+0.3)
    ax.text(1.8, y+0.3, 'No', fontsize=8)
    ax.text(2, y+0.6, '→ Next Epoch', fontsize=8, style='italic')
    
    draw_arrow(5, y, 5, y-0.5)
    ax.text(5.3, y-0.3, 'Yes', fontsize=8)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Training Complete\nLoad Best Model', save_color)
    draw_arrow(5, y, 5, y-0.5)
    
    y -= 1.2
    draw_box(3, y, 4, 0.6, 'Final Evaluation\non Test Set', process_color)
    
    plt.tight_layout()
    plt.savefig('../results/figures/training_pipeline.png', dpi=300, bbox_inches='tight')
    print("Saved: training_pipeline.png")
    plt.close()


def create_comparison_chart():
    """Create model comparison visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = ['Parameters\n(Millions)', 'Training Time\n(Hours)', 
                 'Inference Speed\n(ms)', 'Memory\n(GB)',
                 'Expected Accuracy\n(%)', 'BLEU Score']
    
    baseline_values = [28, 3, 50, 4, 70, 35]
    transformer_values = [340, 10, 200, 12, 80, 50]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Normalize for visualization (except accuracy and BLEU)
    baseline_norm = baseline_values.copy()
    transformer_norm = transformer_values.copy()
    
    bars1 = ax.bar(x - width/2, baseline_norm, width, label='Baseline CNN+LSTM',
                   color='#81C784', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, transformer_norm, width, label='Vision-Language Transformer',
                   color='#64B5F6', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Metrics', fontsize=12, weight='bold')
    ax.set_ylabel('Values', fontsize=12, weight='bold')
    ax.set_title('Model Comparison: Baseline vs Transformer', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('../results/figures', exist_ok=True)
    
    print("Generating architecture diagrams...")
    create_baseline_architecture_diagram()
    create_transformer_architecture_diagram()
    create_training_pipeline_diagram()
    create_comparison_chart()
    print("\nAll diagrams generated successfully!")
    print("Check the 'results/figures/' directory")
