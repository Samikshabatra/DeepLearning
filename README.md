# Lab 4.2: Text Classification using RNN and LSTM

## Overview

This experiment demonstrates the implementation of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks for text classification tasks.

Unlike traditional machine learning models, recurrent neural networks are designed to handle **sequential data**, making them particularly useful for Natural Language Processing (NLP) applications.

In this lab, text data is preprocessed, converted into numerical sequences, and used to train RNN/LSTM models capable of learning patterns and dependencies in textual sequences.

---

## Objectives

• To understand the concept of **sequence modeling**
• To implement **Recurrent Neural Networks (RNN)** for text classification
• To understand the working of **Long Short-Term Memory (LSTM)** networks
• To preprocess textual data for deep learning models
• To evaluate the performance of sequential models on text data

---

## Dataset

The dataset used in this experiment consists of **text samples with corresponding class labels**.
Each text input is processed and converted into sequences before being passed to the neural network.

Typical preprocessing steps include:

* Text cleaning
* Tokenization
* Sequence padding
* Vocabulary creation

---

## Libraries Used

The following Python libraries are used:

* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras
* Scikit-learn

---

## Model Architecture

The implemented models include:

### 1. Recurrent Neural Network (RNN)

Embedding Layer
→ Simple RNN Layer
→ Dense Layer
→ Output Layer

### 2. Long Short-Term Memory (LSTM)

Embedding Layer
→ LSTM Layer
→ Dense Layer
→ Output Layer

LSTM networks improve upon basic RNNs by addressing the **vanishing gradient problem**, allowing them to capture long-term dependencies in text sequences.

---

## Methodology

### 1. Data Preprocessing

* Text cleaning and normalization
* Tokenization using a tokenizer
* Converting text into sequences
* Padding sequences to equal length

### 2. Model Training

The RNN and LSTM models are trained on the processed dataset using:

* Backpropagation through time
* Optimization algorithms such as Adam

### 3. Model Evaluation

The models are evaluated using:

* Training and validation accuracy
* Loss curves
* Prediction results

---

## Results

The LSTM model generally performs better than the basic RNN due to its ability to retain long-term dependencies within sequences.

Performance comparison highlights the advantages of LSTM networks for Natural Language Processing tasks.

---

## Key Concepts Learned

• Sequential data processing
• Tokenization and word embeddings
• Recurrent Neural Networks
• Long Short-Term Memory networks
• Handling vanishing gradient problems
• Text classification using deep learning

---

## Conclusion

This experiment demonstrates how recurrent neural networks can be used to model sequential text data.

The implementation highlights the importance of LSTM networks in capturing long-term dependencies in language, making them widely used in tasks such as sentiment analysis, text classification, and language modeling.

---

## Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning
