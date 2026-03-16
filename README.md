# DeepLearning Repository

This repository contains implementations of **Deep Learning experiments** completed as part of the academic curriculum.
The experiments focus on understanding **neural network fundamentals**, **training mechanisms**, and **performance analysis** using modern deep learning frameworks.

---

# Objectives

* To understand the fundamentals of **Deep Learning and Neural Networks**
* To implement **Multi-Layer Perceptrons (MLP)** and **Deep Feedforward Neural Networks**
* To study the role of **activation functions, loss functions, and optimizers**
* To analyze the impact of **hyperparameters** such as learning rate, batch size, depth, and width
* To gain practical experience using **PyTorch**, **Keras**, **TensorFlow**, and **Scikit-learn**
* To explore advanced architectures including **CNNs, Object Detection, Autoencoders, Transfer Learning, and GANs**

---

# Laboratory Experiments

---

## Lab 1: Learning XOR Boolean Function using MLP

Branch: `Lab1`

Description:

* Implementation of a **Multi-Layer Perceptron (MLP)** to learn the XOR Boolean function
* Demonstrates why XOR is **not linearly separable**
* Highlights the importance of **hidden layers** and **nonlinear activation functions**
* Includes **hyperparameter tuning** and result analysis.

Files:

* `MLP_XOR.ipynb`
* `README.md`

---

## Lab 2: Deep Feedforward Neural Network for Fashion-MNIST Classification

Branch: `Lab2`

Description:

* Design and implementation of a **deep feedforward neural network**
* Classification of **Fashion-MNIST dataset**
* Experiments performed on:

  * Network depth and width
  * Different activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU)
  * Training loss and test accuracy
* Visualization of **hidden layer activations**
* Performance comparison and analysis

---

## Lab 3: Effect of Regularization Techniques on Model Training

Branch: `Lab3`

Description:

* Implementation and comparison of **L1 (Lasso), L2 (Ridge), and Elastic Net Regularization**
* Demonstrates how regularization helps prevent **overfitting**
* Analysis of model performance using:

  * **Mean Squared Error (MSE)**
  * **R² Score**
* Understanding how regularization affects **model coefficients and generalization**

Concepts Covered:

* Ridge Regression
* Lasso Regression
* Elastic Net Regression
* Model regularization
* Bias-Variance tradeoff

---

## Lab 4: CNN and YOLO for Image Classification and Object Detection

Branch: `Lab4`

Description:

* Implementation of a **Convolutional Neural Network (CNN)** for image classification

* Training the model on image datasets such as **CIFAR-10**

* Understanding the role of CNN components including:

  * Convolution layers
  * Activation functions
  * Pooling layers
  * Batch Normalization
  * Dropout

* Implementation of **YOLO (You Only Look Once)** for **real-time object detection**

* Detection of multiple objects within a single image using bounding boxes

* Visualization of detected objects with class labels and confidence scores

Concepts Covered:

* Convolutional Neural Networks (CNN)
* Feature extraction in images
* Object detection techniques
* YOLO architecture
* Bounding box prediction
* Real-time detection systems

---

## Lab 5: Autoencoder for Feature Learning and Dimensionality Reduction

Branch: `Lab5`

Description:

* Implementation of an **Autoencoder neural network**
* Learning compressed representations of input data
* Reconstruction of original data from latent representations
* Evaluation using **reconstruction error**

Concepts Covered:

* Encoder–Decoder architecture
* Latent space representation
* Dimensionality reduction
* Unsupervised learning

---

## Lab 6: Image Classification using Transfer Learning

Branch: `Lab6`

Description:

* Implementation of **Transfer Learning** using a pre-trained CNN model
* Leveraging knowledge from large pre-trained models such as **VGG16 / ResNet / MobileNet**
* Fine-tuning the model for a new dataset
* Performance evaluation using training and validation metrics

Concepts Covered:

* Transfer learning
* Feature extraction
* Fine-tuning deep neural networks
* Efficient deep learning training

---

## Lab 7: Generative Adversarial Networks (GAN) for Synthetic Data Generation

Branch: `Lab7`

Description:

* Implementation of **Generative Adversarial Networks (GANs)**
* Training two competing neural networks:

  * **Generator** – produces synthetic samples
  * **Discriminator** – distinguishes real vs fake samples
* Demonstrates adversarial training and synthetic data generation

Concepts Covered:

* Generative models
* Adversarial training
* Synthetic data generation
* Deep generative networks

---

# Technologies Used

* Python
* PyTorch
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook / Google Colab

---

# Key Concepts Covered

* Feedforward Neural Networks
* Multi-Layer Perceptron (MLP)
* Backpropagation Algorithm
* Gradient Descent Optimization
* Activation Functions
* Convolutional Neural Networks (CNN)
* Object Detection (YOLO)
* Autoencoders
* Transfer Learning
* Generative Adversarial Networks (GAN)
* Hyperparameter Tuning
* Model Evaluation and Visualization

---

# Learning Outcomes

Through these experiments, I gained:

* Hands-on experience with **deep learning model implementation**
* Understanding of **neural network architectures**
* Ability to analyze **training behavior and model performance**
* Practical exposure to **PyTorch, TensorFlow, and Keras frameworks**
* Experience with **computer vision, generative models, and representation learning**

---

# Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning
Deep Learning Academic Repository
