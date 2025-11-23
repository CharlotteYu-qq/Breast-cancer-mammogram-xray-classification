# Breast Cancer Mammography Classification:

A deep learning system for multi-class classification of breast cancer pathology from mammography images using the CBIS-DDSM dataset.


# Project Overview:

    This project develops an automated deep learning system for classifying breast lesions in mammography images into three pathological categories:

    BENIGN - Non-cancerous abnormalities
    MALIGNANT - Cancerous lesions
    BENIGN_WITHOUT_CALLBACK - Benign findings not requiring follow-up

# Key Features:

    Multi-class Classification: 3-class pathology prediction

    Medical Image Processing: DICOM-specific preprocessing with window-level adjustment

    Deep Learning Models: ResNet18/34/50, EfficientNet b0/b1 architectures
    
    Rigorous Validation: 5-fold cross-validation with patient-wise splitting

    Class Imbalance Handling: Weighted loss functions and sampling strategies

    Comprehensive Evaluation: Balanced accuracy, confusion matrices, precision, recall

# Dataset:

    CBIS-DDSM (Curated Breast Imaging Subset)
    2,785 curated full-field mammograms
    Expert-annotated pathology labels
    Standard CC and MLO views
    Calcification and mass abnormalities

# Data Distribution:

    MALIGNANT: 1,294 cases (46.5%)
    BENIGN: 1,219 cases (43.8%)
    BENIGN_WITHOUT_CALLBACK: 272 cases (9.8%)


# Installation:

    Prerequisites
        Python 3.8+
        PyTorch 1.12+
        CUDA-capable GPU (recommended)
        
    Dependencies
        pip3 install torch torchvision torchaudio
        pip3 install pydicom pandas scikit-learn matplotlib
        pip3 install pillow opencv-python numpy


# Quick Start
1. Data Preparation

    Download CBIS-DDSM dataset from TCIA
    
    Place DICOM files in ./dicom_data/
    Place CSV annotations in ./csv/

2. Preprocessing

    python3 create_strict_unique_full_mammo.py
    
    python3 create_csv.py
    
    This creates the training/validation splits and preprocessed metadata.

3. Training & evaluation

    a. Train with default parameters:
    
    python3 main_breast.py --backbone resnet18 --batch_size 16 --lr 1e-3 --epochs 50

    b. Train with specific configuration:

    python3 main_breast.py \
    --backbone resnet50 \
    --batch_size 48 \
    --lr 5e-5 \
    --epochs 35 \
    --weight_decay 2e-4

# Project Structure

    breast-cancer-classification/
    ├── main_breast.py # Main training script
    ├── trainer.py # Training and validation logic
    ├── model.py # Model architecture
    ├── dataset.py # Data loading and preprocessing
    ├── args.py # Configuration arguments
    ├── utils.py # Utility functions (plotting, etc.)
    ├── requirements.txt # Python dependencies
    └── README.md # This file




