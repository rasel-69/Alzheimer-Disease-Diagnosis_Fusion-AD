# Fusion-AD: Fuzzy Attention Enhanced Few Shot Learning For Alzheimer Disease Diagnosis
# OverView : 
 We integrates FMSAM
and Gated SwinViT to improve encoder feature expression by collecting multi scale spatial structures also efficiently modeling
uncertainty using fuzzy learning techniques. Fuzzy Gated Attention Module (FGAM) is incorporated to alter the traditional shifted
window multi head attention in SwinViT, allowing long range contextual designing with gated and noise reduced refinement.
Moreover to enhance feature separation the Mish activation is integrated into the Swin Transformer MLP modules in place of
GELU, ensuring the framework to generalize efficiently under limited training samples. Robust assessments executed along with
Grad-CAM and LIME on two Alzheimer disease datasets exhibits that our proposed model achieved a accuracy of 96.00%, a
precision of 97.00%, a recall of 96.00% and a f1-score of 97.00% on the K1 dataset. It secured accuracy of 94.45%, precision of
94.04%, recall of 95.28%, and a f1-score of 94.04% on the K2 dataset.


# Key Features

## Few-Shot Learning (FSL)
#### Implements Prototypical Networks for N-way K-shot learning.
#### Permits Alzheimer disease diagnosis with Limited sample Images.
### Fuzyy Multi Scale Attention Module (FMSAM) In the Encoder.
### Added Fuzzy Gated Attention Module based SwinViT Transformer.


# Technologies Used

#### PyTorch: Deep learning framework for model building and training.
#### Torchvision: For image transforms and preprocessing.
#### PIL : For image reading
#### NumPy: Data manipulation.
#### Python 3.10+: Programming language.
#### GPU Acceleration (CUDA): Optional, for faster training.
#### Mixed-Precision Training (torch.cuda.amp): For efficient GPU usage.
#### Used Google Colab

# Data 
### K1 Dataset (oasis dataset From Kaggle):
#### Link: https://www.kaggle.com/datasets/raselahmed2091/oasis-dataset

### K2 Dataset (adni dataset From kaggle): 
#### Link: https://www.kaggle.com/datasets/raselahmed2091/adni-dataset






