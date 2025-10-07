XAI-Driven Multimodal Learning and Conditional Attention WGAN

NeurIPS 2025 Reproducible Implementation

Overview

This repository provides a reproducible implementation of a multimodal learning framework that integrates vision and language, augmented with generative modeling, bias mitigation, and explainability. The project implements the algorithms and evaluation metrics described in the NeurIPS 2025 paper, including Algorithm 2 (multimodal classification with attention fusion) and Algorithms 1 and 3 (conditional attention WGAN with gradient penalty and bias regularization). The framework is designed to handle datasets combining images, text, and protected attributes, providing both high performance and fairness-aware predictions.

Our contributions can be summarized as follows:

Multimodal Classifier: Combines a ResNet-50 image encoder with a BERT-base text encoder using a cross-modal attention fusion module. This architecture allows the network to leverage complementary information from visual and textual modalities for classification tasks.

Conditional Attention WGAN: Implements a generator and critic network capable of conditional image synthesis with gradient penalty for stability. A bias regularizer aligns the distribution of generated samples across protected groups, mitigating undesired societal biases in downstream applications.

Explainability & Feedback Loop: Integrates Grad-CAM++ for visual attribution, LIME/weighted surrogate models for local explanations, and a Reveal-to-Revise iterative loop to refine predictions and reduce model bias in sensitive attributes.

Evaluation Suite: Provides metrics for predictive performance (Accuracy, F1), generative quality (SSIM, FID), and fairness (∆bias), as well as IoU alignment between model explanations and ground truth regions.

Directory Structure
xai_project/
├─ data/                 # datasets (images, texts, labels, protected attributes)
├─ src/
│  ├─ models.py           # image & text encoders + classifier head
│  ├─ fusion.py           # cross-modal attention fusion
│  ├─ wgan.py             # conditional generator & critic
│  ├─ train_classifier.py # multimodal classifier training loop
│  ├─ train_wgan.py       # conditional WGAN training loop
│  ├─ explain.py          # Grad-CAM++, LIME, Reveal-to-Revise
│  └─ eval.py             # evaluation metrics & visualization
├─ configs/
│  └─ default.yaml        # hyperparameters and experimental setup
├─ requirements.txt
└─ run.sh                 # full pipeline execution script


This structure allows reproducibility and modularity. All experiments, including ablations and hyperparameter sweeps, can be reproduced by modifying the YAML configuration and running the respective training scripts.

Environment Setup

We recommend Python 3.9–3.11 with GPU-enabled PyTorch. To replicate the results:

# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Core libraries
pip install torch torchvision torchaudio       # select CUDA version
pip install transformers timm datasets
pip install scikit-learn pandas numpy tqdm
pip install tensorboard matplotlib pillow
pip install pytorch-grad-cam lime shap wandb


This environment ensures compatibility with ResNet-50, BERT-base, and generative modules.

Data Preparation

The project supports multimodal datasets. Each sample consists of:

Image (I)

Text (T)

Label (Y)

Protected Attribute (B)

Recommended datasets:

Text-only baseline: Jigsaw Toxic Comments (Kaggle)

Image + Text: MS-COCO (captions) or domain-specific datasets (e.g., medical images + reports)

Preprocessing steps:

Images: Resize to 224×224 and normalize using ImageNet mean/std.

Text: Tokenize using BERT tokenizer, pad/truncate to 128 tokens.

Split: Stratified Train/Validation/Test (80/20) to maintain label and protected attribute balance.

Model Architecture
1. Multimodal Classifier

Image Encoder: ResNet-50 backbone with final fully connected layer removed and projected to 512-dimensional feature space.

Text Encoder: BERT-base with pooler output projected to 512-dimensional embedding.

Cross-Modal Attention Fusion: Multi-head attention module integrating visual and textual embeddings, producing a joint representation.

Classifier Head: LayerNorm, dropout, and MLP layers for final classification.

2. Conditional Attention WGAN

Generator: Receives latent vector z concatenated with one-hot encoded conditional labels. Outputs synthesized images.

Critic: Convolutional network augmented with label-conditioning through channel-wise concatenation. Produces real/fake score.

Training Stabilization: Gradient penalty is applied on interpolated samples between real and generated images.

Bias Regularizer: Computes the L2 difference between the expectations of protected attributes in generated and real samples, reducing bias in generation.

Training Pipelines
Multimodal Classifier

Loss Function: Cross-entropy or BCE.

Bias Penalty:

R_bias = || E[B(preds)] - E[B(labels)] ||^2


Optimizer: AdamW with cosine annealing scheduler.

Training Loop: Forward pass through encoders → fusion → classifier; backward pass includes optional bias regularization.

Conditional Attention WGAN

Critic Updates: n_critic updates per generator update.

Gradient Penalty: Ensures Lipschitz constraint.

Generator Updates: Minimize negative critic score + bias regularizer.

Optional Attribute Extraction: Pre-trained attribute classifier used for computing bias on generator outputs.

Explainability & Feedback

Grad-CAM++: Generates visual attribution heatmaps over convolutional layers.

LIME / Weighted Surrogate Models: Creates local perturbations and fits linear models weighted by similarity kernels to explain predictions.

Reveal-to-Revise: Iteratively adjusts model predictions and retrains with attention on biased regions to improve fairness.

Example Grad-CAM++ snippet:

from pytorch_grad_cam import GradCAM
cam = GradCAM(model=img_enc.back, target_layer=img_enc.back.layer4[-1], use_cuda=True)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

Evaluation

Metrics cover performance, fairness, and explanation quality:

Classification: Accuracy, F1-score

Generative: SSIM (structural similarity), FID (distributional realism)

Fairness: ∆bias across protected groups

Explanation Alignment: IoU-XAI between attributions and ground truth regions

Logging via TensorBoard or WandB supports metric tracking, loss curves, and visualization grids combining real, generated, and attribution images.

Reproducibility

Set seeds and enforce deterministic CUDNN behavior.

Save full YAML configuration and git commit hash.

Save checkpoints of top-performing models and evaluation metrics.

Repeat experiments at least three times for robustness.

Ablation Studies

The repository supports systematic ablations to quantify contributions:

Single-modality baselines: Image-only or text-only.

Without fusion module: Disabling attention fusion.

Without XAI module: Turning off Grad-CAM++ and LIME explanations.

Without bias feedback: Disable Reveal-to-Revise loop.

Hyperparameter sweeps can be conducted using Optuna or Ray Tune, targeting accuracy, SSIM, and ∆bias.

Quick Start
# Activate environment
source venv/bin/activate

# Train classifier
python src/train_classifier.py --config configs/default.yaml

# Train WGAN
python src/train_wgan.py --config configs/default.yaml

# Evaluate models
python src/eval.py --ckpt models/best_classifier.pth --data data/

# Produce explainability outputs
python src/explain.py --ckpt models/best_classifier.pth --output visualizations/

Deliverables

models/ — trained weights (.pt, .pth)

results/metrics.json — evaluation metrics per run

visualizations/ — Grad-CAM overlays, surrogate explanations, generated samples

notebooks/ — ablation studies and analysis

Citation
@inproceedings{yourpaper2025,
  title={Title of Your NeurIPS 2025 Paper},
  author={Your Name et al.},
  booktitle={NeurIPS},
  year={2025}
}
