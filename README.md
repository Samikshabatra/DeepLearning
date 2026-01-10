# DeepLearning: 
# Learning XOR Boolean Function using MLP:
This repository contains the implementation of Lab 1 of the Deep Learning Laboratory, where a Multi-Layer Perceptron (MLP) is designed and trained to learn the XOR Boolean function using three popular deep learning frameworks: Keras (TensorFlow), PyTorch, and TensorFlow Low-Level API.

The XOR problem is a classic example of a non-linearly separable dataset, making it an ideal case to demonstrate the need for hidden layers and nonlinear activation functions in neural networks.

# Objective:

The objective of this experiment is to:

Implement an MLP to solve the XOR problem

Compare implementations across Keras, PyTorch, and TensorFlow

Study the effect of key hyperparameters such as learning rate, number of neurons, and number of epochs

# Dataset:

The XOR truth table used:

| Input 1 | Input 2 | Output |
| ------- | ------- | ------ |
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

# Model Architecture:

Each implementation uses a similar MLP structure:

Input Layer: 2 neurons

Hidden Layer: 2, 4, and 8 neurons (tested)

Activation Function: ReLU / Tanh

Output Layer: 1 neuron

Output Activation: Sigmoid

Loss Function: Binary Cross-Entropy

Optimizer: Adam

# Hyperparameter Tuning:

To improve model performance, hit-and-trial experimentation was performed using:

Hidden neurons: 2, 4, 8

Learning rates: 0.001, 0.01, 0.08, 0.3

Epochs: 500 and 2000

The best configurations achieved 100% accuracy (4/4 correct predictions) for the XOR function in all three frameworks.

# Repository Contents:

This branch contains:

MLP_XOR.ipynb – Complete implementation and experiments

Lab1_DL.pdf – Lab manual with code, results, and conclusion

# Result:

All three deep learning frameworks successfully learned the XOR Boolean function when proper hyperparameters and hidden layers were used, confirming the effectiveness of MLPs in solving non-linear classification problems.

Name: Samiksha Batra

Program: M.Sc. Artificial Intelligence & Machine Learning
