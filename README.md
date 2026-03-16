# Lab 6: Image Classification using Transfer Learning

## Overview

This experiment demonstrates the use of **Transfer Learning** for image classification using a pre-trained Convolutional Neural Network (CNN).

Training deep learning models from scratch requires large datasets and significant computational resources. Transfer learning solves this problem by using pre-trained models that have already learned useful features from large datasets such as ImageNet.

In this lab, a pre-trained CNN model is used as a feature extractor and fine-tuned for a new image classification task.

---

## Objectives

• To understand the concept of **Transfer Learning**
• To utilize **pre-trained CNN models** for image classification
• To perform **feature extraction using deep neural networks**
• To fine-tune a pre-trained model for a specific dataset
• To evaluate model performance in computer vision tasks

---

## Transfer Learning Concept

Transfer learning allows a model trained on one task to be reused for another related task.

Instead of training a deep neural network from scratch, we reuse the learned weights of a pre-trained network and adapt it for a new dataset.

Advantages of transfer learning:

* Faster training
* Better performance with small datasets
* Reduced computational cost
* Improved feature extraction

---

## Pre-trained Models

Common CNN architectures used for transfer learning include:

* VGG16
* ResNet
* MobileNet
* Inception

These models are trained on the **ImageNet dataset**, which contains millions of labeled images across thousands of categories.

---

## Libraries Used

The following Python libraries are used in this experiment:

* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras
* Scikit-learn

---

## Methodology

### 1. Data Preparation

* Load image dataset
* Resize images to match model input size
* Normalize pixel values
* Split dataset into training and testing sets

### 2. Model Selection

A **pre-trained CNN model** is loaded with pre-trained weights.

The convolutional base of the network is used as a **feature extractor**.

### 3. Model Modification

* The top classification layers of the model are removed
* New fully connected layers are added
* The model is adapted to the new classification task

### 4. Model Training

The model is trained on the new dataset using transfer learning techniques.

Training includes:

* Freezing base layers
* Fine-tuning selected layers
* Optimizing using the Adam optimizer

### 5. Model Evaluation

The trained model is evaluated using:

* Training accuracy
* Validation accuracy
* Loss curves

---

## Results

Transfer learning significantly improves model performance and reduces training time compared to training a CNN from scratch.

The model successfully learns useful image features from the dataset and achieves good classification performance.

Training and validation curves demonstrate stable learning behavior.

---

## Key Concepts Learned

• Transfer learning in deep learning
• Pre-trained CNN architectures
• Feature extraction using convolutional networks
• Fine-tuning deep neural networks
• Image classification using deep learning

---

## Applications of Transfer Learning

• Medical image analysis
• Object detection
• Face recognition
• Autonomous driving
• Image classification systems

---

## Conclusion

This experiment demonstrates how transfer learning can efficiently adapt powerful pre-trained deep learning models for new image classification tasks.

By leveraging previously learned features, transfer learning enables faster training and improved performance, making it one of the most widely used techniques in modern computer vision applications.

---

## Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning
