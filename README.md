# Lab 7: Generative Adversarial Networks (GAN)

## Overview

This experiment demonstrates the implementation of **Generative Adversarial Networks (GANs)**, a powerful deep learning architecture used for generating new synthetic data that resembles real data.

GANs consist of two competing neural networks:

• **Generator** – Generates synthetic data
• **Discriminator** – Distinguishes between real and generated data

Both networks are trained simultaneously in an adversarial process where the generator tries to fool the discriminator while the discriminator tries to correctly identify real and fake samples.

---

## Objectives

• To understand the architecture of **Generative Adversarial Networks**
• To implement **Generator and Discriminator networks**
• To train GANs using adversarial training
• To generate synthetic data samples
• To analyze how GANs learn complex data distributions

---

## GAN Architecture

A Generative Adversarial Network consists of two main components:

### 1. Generator

The generator network creates synthetic data from random noise.

Input: Random noise vector
→ Fully connected / neural layers
→ Output: Generated sample

The objective of the generator is to produce data that looks similar to real data.

---

### 2. Discriminator

The discriminator is a binary classifier that distinguishes between real and generated data.

Input: Real or generated sample
→ Neural network layers
→ Output: Probability (Real / Fake)

The goal of the discriminator is to correctly identify whether a sample is real or fake.

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

* Load the dataset
* Normalize input data
* Prepare training batches

### 2. Model Construction

Two neural networks are defined:

Generator Network:

* Takes random noise as input
* Produces synthetic samples

Discriminator Network:

* Takes real or generated samples
* Classifies them as real or fake

### 3. Adversarial Training

The GAN is trained through an adversarial process:

1. The generator creates synthetic samples from random noise.
2. The discriminator receives both real and generated samples.
3. The discriminator learns to classify real vs fake.
4. The generator improves by trying to fool the discriminator.

This process continues iteratively until the generated samples become realistic.

---

## Results

During training:

* The generator gradually improves its ability to generate realistic samples.
* The discriminator becomes better at detecting fake data.
* Over time, the generator learns the underlying distribution of the dataset.

Training loss curves and generated samples demonstrate the progress of the GAN model.

---

## Key Concepts Learned

• Generative models in deep learning
• Adversarial training
• Generator and discriminator networks
• Synthetic data generation
• Learning complex data distributions

---

## Applications of GANs

• Image generation
• Data augmentation
• Style transfer
• Deepfake generation
• Super-resolution imaging

---

## Conclusion

This experiment demonstrates how Generative Adversarial Networks can learn to generate realistic synthetic data through adversarial training.

GANs represent one of the most powerful generative models in deep learning and have wide applications in computer vision, data generation, and creative AI systems.

---

## Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning
