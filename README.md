# Lab 4: CNN from Scratch for CIFAR-10 Image Classification

## Overview
This experiment demonstrates the implementation of a Convolutional Neural Network (CNN) from scratch using NumPy for image classification tasks. The model is trained and evaluated using the CIFAR-10 dataset.

The goal of this lab is to understand the internal workings of convolutional neural networks by manually implementing the core building blocks such as convolution layers, pooling layers, activation functions, and fully connected layers.

---

## Objectives

• To understand the architecture of Convolutional Neural Networks (CNNs)  
• To implement convolution and pooling operations from scratch  
• To train a CNN model for image classification  
• To evaluate model performance using training metrics  
• To gain deeper insight into how CNNs process image data  

---

## Dataset

The experiment uses the **CIFAR-10 dataset**, which contains 60,000 images across 10 different classes.

Each image has a size of **32 × 32 pixels with 3 color channels (RGB).**

The dataset includes the following classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Libraries Used

The following Python libraries are used in this experiment:

- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)
- KaggleHub (for dataset download)

---

## CNN Architecture

The implemented CNN architecture consists of the following layers:

Conv(3 → 32, 3×3) → BatchNorm → ReLU  
Conv(32 → 32, 3×3) → BatchNorm → ReLU  
MaxPooling → Dropout  

Conv(32 → 64, 3×3) → BatchNorm → ReLU  
Conv(64 → 64, 3×3) → BatchNorm → ReLU  
MaxPooling → Dropout  

Conv(64 → 128, 3×3) → BatchNorm → ReLU  
MaxPooling → Dropout  

Flatten Layer  

Fully Connected Layer (2048 → 512)  
BatchNorm → ReLU → Dropout  

Output Layer (512 → 10)

---

## Methodology

### 1. Data Loading
The CIFAR-10 dataset is downloaded and images are loaded into memory for training and testing.

### 2. Data Preprocessing
Images are resized and normalized before being fed into the CNN model.

### 3. Model Implementation
A convolutional neural network is implemented from scratch using NumPy, including:

- Convolution layers
- ReLU activation
- Batch Normalization
- Max Pooling
- Dropout
- Fully Connected Layers

### 4. Model Training
The model is trained using mini-batch gradient descent with multiple epochs.

### 5. Model Evaluation
The performance of the CNN is evaluated using training loss and prediction accuracy.

---

## Results

The CNN model successfully learns image features through convolution operations and gradually reduces training loss during epochs.

Visualizations include:

- Training loss curve
- Prediction outputs
- Model performance plots

---

## Key Concepts Learned

• Convolution operations in CNN  
• Feature extraction from images  
• Pooling and dimensionality reduction  
• Regularization using dropout  
• Batch normalization for stable training  
• End-to-end CNN training pipeline  

---

## Conclusion

This experiment demonstrates how Convolutional Neural Networks work internally by implementing each component from scratch.

The lab provides a strong conceptual understanding of how deep learning models process image data and how CNN architectures are designed for computer vision tasks.

---

## Author

Samiksha Batra  
MSc Artificial Intelligence & Machine Learning
