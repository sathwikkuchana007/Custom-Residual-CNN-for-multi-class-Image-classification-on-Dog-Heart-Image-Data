Custom Residual CNN for Multi-Class Image Classification on Dog Heart Data
This repository contains the implementation of a custom convolutional neural network (CNN) for multi-class classification of dog heart X-ray images. This work is based on the research paper "Custom Residual CNN for Multi-Class Image Classification on Dog Heart Data" by Sathwik Kuchana.

The project focuses on building a computationally efficient yet accurate model for classifying X-ray images into three categories: Small, Normal, and Large hearts. The model addresses challenges such as limited data, overfitting, and computational constraints in resource-constrained environments like veterinary clinics.

Table of Contents
Introduction
Features
Dataset
Model Architecture
Installation
Usage
Training
Evaluation
Results
Future Work
References
Contact
Introduction
Veterinary diagnostics often require accurate classification of radiographs for conditions like cardiomegaly. Manual methods, while effective, are prone to human error and time-consuming. This project introduces a lightweight CNN architecture that integrates advanced techniques like residual connections, spatial attention, and DropBlock regularization to address these challenges and ensure robust performance.

Features
Residual Connections: Mitigates vanishing gradient problems, enabling deeper networks.
Spatial Attention Mechanism: Focuses on clinically significant regions in the X-ray images.
DropBlock Regularization: Reduces overfitting, ensuring better generalization.
Lightweight Design: Optimized for deployment in low-resource environments.
Dataset
The dataset consists of 2,000 X-ray images classified into:

Small Heart
Normal Heart
Large Heart
Preprocessing
Resizing: All images resized to 224x224 pixels.
Normalization: Pixel values normalized based on ImageNet statistics.
Data Augmentation: Includes cropping, flipping, and rotation to enhance dataset variability.
Model Architecture
The custom model, ResNetDogHeart, is inspired by ResNet and tailored for efficient multi-class classification.

Key Components
Convolutional Layer: Initial feature extraction.
Residual Bottleneck Blocks: Facilitates deeper feature extraction and gradient flow.
Global Average Pooling (GAP): Reduces spatial dimensions for efficient feature representation.
Fully Connected Layer: Maps features to class probabilities.
Installation
Prerequisites
Python 3.7 or later
PyTorch
NumPy
Matplotlib
Steps
Clone this repository:
bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Place the dataset in the data/ directory.
Usage
Training
To train the model, run:

bash
Copy code
python train.py
Evaluation
To evaluate the model, run:

bash
Copy code
python evaluate.py --model_path saved_models/resnet_dog_heart.pth
Results
Performance Metrics
Test Accuracy: 69.5%
Validation Accuracy: 67.0%
Comparison with State-of-the-Art
Metric	Proposed Model	Regressive Vision Transformer (RVT)
Validation Accuracy (%)	67.0	85.0
Test Accuracy (%)	69.5	87.3
Parameters (Millions)	12.8	19.6
Highlights
Lightweight architecture for low-resource environments.
Robust performance despite limited training data.
Practical for deployment in veterinary clinics.
Future Work
Synthetic Data Generation: Use GANs to generate additional X-ray images for underrepresented classes.
Hybrid Architectures: Combine CNNs and transformers for enhanced feature extraction.
Explainability Tools: Integrate Grad-CAM for interpretability in clinical use.
Semi-Supervised Learning: Leverage unlabeled data to reduce manual labeling efforts.
References
He et al., 2016: Deep Residual Learning for Image Recognition
Ghiasi et al., 2018: DropBlock: A Regularization Method for Convolutional Networks
Dosovitskiy et al., 2021: An Image is Worth 16x16 Words: Transformers for Image Recognition
For detailed insights, refer to the research paper.

Contact
For questions or collaboration:

Author: Sathwik Kuchana
Email: skuchana@mail.yu.edu
Institution: Yeshiva University
