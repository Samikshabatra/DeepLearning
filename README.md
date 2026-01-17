# Lab 2: Deep Feedforward Neural Network for Fashion-MNIST Classification

## Aim
To design, implement, and evaluate a **Deep Feedforward Neural Network** for classifying images from the **Fashion-MNIST dataset** using **PyTorch**, and to understand the role of **network depth**, **nonlinear activation functions**, **loss computation**, and **gradient-based learning** in a supervised classification task.

---

## Dataset
**Fashion-MNIST** is a dataset of Zalando’s article images consisting of:
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 grayscale
- 10 clothing categories

Each image belongs to one of the following classes:

- T-shirt/top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle boot  

---

## Objectives
- To load and preprocess the Fashion-MNIST dataset
- To build a **deep feedforward neural network** with multiple hidden layers
- To study the effect of **ReLU and other activation functions**
- To train the model using **Cross-Entropy Loss** and **Adam optimizer**
- To monitor training loss and accuracy
- To evaluate model performance on unseen test data
- To analyze the impact of **hyperparameters**
- To visualize hidden layer activations

---

## Model Architecture
- Input Layer: 784 neurons (28 × 28 flattened image)
- Hidden Layers:  
  - 3 or more fully connected layers  
  - Configurable depth and width
- Activation Functions:
  - ReLU (primary)
  - Sigmoid
  - Tanh
  - Leaky ReLU
- Output Layer:
  - 10 neurons (one for each class)
  - Linear activation (logits)

---

## Training Details
- Loss Function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Learning Rate: 0.001
- Batch Size: 64
- Number of Epochs: 10 (base experiment)

---

## Methodology

### 1. Forward Pass
The input image is flattened and passed through multiple hidden layers where weighted sums and nonlinear activations are applied to generate output logits.

### 2. Loss Computation
The predicted logits are compared with true labels using **cross-entropy loss**, which measures classification error.

### 3. Backward Pass
Gradients of the loss with respect to model parameters are computed using **backpropagation**.

### 4. Gradient Update
The optimizer updates the weights using gradient descent to minimize the loss.

---

## Experiments Performed

### 🔹 Experiment 1: Network Depth and Width
- Hidden layers tested: 1, 3, and 5
- Neurons per layer tested: 64 and 128
- Observed trade-off between **model capacity and overfitting**

### 🔹 Experiment 2: Activation Function Comparison
- ReLU
- Sigmoid
- Tanh
- Leaky ReLU

ReLU and Leaky ReLU showed faster convergence and better accuracy due to reduced vanishing gradient issues.

### 🔹 Experiment 3: Hidden Layer Visualization
- Forward hooks used to capture hidden layer activations
- Activations plotted to observe how features are transformed layer by layer
- Early layers capture low-level patterns, deeper layers capture abstract representations

---

## Results
- Training Accuracy: ~90%
- Test Accuracy: ~88–91%
- ReLU and Leaky ReLU performed better than Sigmoid and Tanh
- Deeper networks improved representation but increased overfitting risk

---

## Key Observations
- Increasing depth improves feature learning but may lead to overfitting
- ReLU-based activations perform best for deep networks
- Proper choice of hyperparameters significantly affects convergence
- Visualization helps understand internal feature transformations

---

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook / Google Colab

---

## Conclusion
In this experiment, a deep feedforward neural network was successfully implemented to classify Fashion-MNIST images. The study demonstrated the importance of network depth, nonlinear activations, and gradient-based optimization. The experiments showed that ReLU-based deep networks provide better convergence and generalization performance for image classification tasks.

---

## Author
**Samiksha Batra**  
Deep Learning   
Lab 2 – Fashion-MNIST Classification
