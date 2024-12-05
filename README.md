# Custom-Residual-CNN-for-multi-class-Image-classification-on-Dog-Heart-Image-Data
Custom Residual CNN for multi class Image classification on Dog Heart Image Data


# Custom Residual CNN for Multi-Class Image Classification on Dog Heart Data

This repository contains the implementation and research code for the paper **"Custom Residual CNN for Multi-Class Image Classification on Dog Heart Data"** by Sathwik Kuchana. The study focuses on a custom convolutional neural network (CNN) architecture tailored for multi-class classification of dog heart X-ray images into three categories: **small**, **normal**, and **large** hearts. The model is designed for computational efficiency and accuracy, particularly in resource-constrained veterinary settings.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Accurate classification of dog heart radiographs into clinically relevant categories plays a critical role in diagnosing conditions like cardiomegaly. This repository provides the code and techniques used to develop a lightweight yet effective CNN model incorporating:

- **Residual Blocks**: To address vanishing gradient issues.
- **Spatial Attention Mechanisms**: To focus on significant image regions.
- **DropBlock Regularization**: To prevent overfitting.

---

## Dataset

The dataset consists of **2,000 X-ray images** categorized into three classes:
1. **Small Heart**
2. **Normal Heart**
3. **Large Heart**

### Preprocessing Steps
- **Resizing**: All images resized to `224x224` pixels.
- **Normalization**: Pixel values normalized based on ImageNet statistics.
- **Data Augmentation**: Includes random cropping, flipping, and rotation to enhance dataset variability.

---

## Model Architecture

The architecture, named `ResNetDogHeart`, is inspired by ResNet and features:
- **Bottleneck Residual Blocks** for efficient gradient flow.
- **Global Average Pooling (GAP)** for dimensionality reduction.
- **Softmax Layer** for classification into three classes.

### Key Components
1. **Convolutional Layer**: Initial feature extraction.
2. **Residual Layers**: Hierarchical feature learning.
3. **GAP Layer**: Reduction of spatial dimensions.
4. **Fully Connected Layer**: Mapping features to class probabilities.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
pip install -r requirements.txt

Usage
Training the Model
Prepare the dataset by organizing it into train/, val/, and test/ folders.
Run the training script:
bash
Copy code
python train.py
Evaluating the Model
Use the evaluation script to test the trained model:

bash
Copy code
python evaluate.py --model_path saved_models/resnet_dog_heart.pth
Results
Model Performance
Test Accuracy: 69.5%
Validation Accuracy: 67.0%
Key Features
Residual Connections: Efficient training for deep networks.
Data Augmentation: Enhanced robustness.
Lightweight Design: Suitable for deployment in resource-constrained environments.

#Future Work
Improved Dataset Diversity: Use GANs for synthetic data generation.
Explainability Tools: Integrate Grad-CAM for interpretability.
Hybrid Architectures: Combine CNNs with transformers for enhanced accuracy.
Semi-Supervised Learning: Leverage unlabeled data for improved performance.

#References
Deep Residual Learning for Image Recognition
DropBlock: A Regularization Method for CNNs
An Image is Worth 16x16 Words: Transformers for Image Recognition
For a detailed discussion, refer to the research paper.

Contact
For questions or collaboration:

Author: Sathwik Kuchana
Email: skuchana@mail.yu.edu
Institution: Yeshiva University
