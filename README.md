# Lab 4.2: Object Detection using YOLO

## Overview

This experiment demonstrates **object detection using the YOLO (You Only Look Once) algorithm**, one of the most efficient real-time object detection models in computer vision.

Unlike traditional object detection methods that scan images multiple times, YOLO processes the entire image in a single forward pass through a neural network. This enables fast and accurate detection of multiple objects within an image.

In this lab, YOLO is used to detect and classify objects in images by predicting **bounding boxes, class labels, and confidence scores**.

---

## Objectives

* To understand the concept of **object detection in computer vision**
* To implement **YOLO (You Only Look Once) object detection**
* To detect multiple objects within an image
* To visualize bounding boxes around detected objects
* To analyze detection confidence and classification results

---

## YOLO Algorithm

YOLO is a **single-stage object detection algorithm** that divides an image into a grid and predicts bounding boxes and class probabilities for each grid cell.

Key components of YOLO include:

* **Bounding Box Prediction**
  Determines the location of detected objects in the image.

* **Confidence Score**
  Indicates how confident the model is about the presence of an object.

* **Class Probability**
  Determines which object class is detected.

YOLO performs detection in **one single neural network pass**, making it significantly faster than traditional object detection techniques.

---

## Libraries Used

The following Python libraries are used in this experiment:

* Python
* OpenCV
* NumPy
* Matplotlib
* PyTorch / YOLO framework

---

## Methodology

### 1. Image Loading

The input image is loaded using OpenCV and prepared for processing.

### 2. Model Initialization

A pre-trained **YOLO object detection model** is loaded.

### 3. Object Detection

The model processes the image and predicts:

* Bounding boxes
* Class labels
* Confidence scores

### 4. Visualization

Detected objects are displayed by drawing **bounding boxes and labels** around them in the image.

---

## Results

The YOLO model successfully detects objects present in the image and highlights them using bounding boxes.

Each detected object is labeled with:

* Object class name
* Detection confidence score

The results demonstrate the capability of YOLO to perform **fast and accurate real-time object detection**.

---

## Key Concepts Learned

* Object detection in computer vision
* YOLO architecture
* Bounding box prediction
* Confidence score calculation
* Real-time detection systems
* Visualization of detection results

---

## Applications of YOLO

* Autonomous driving
* Surveillance systems
* Face detection
* Traffic monitoring
* Retail analytics
* Robotics and automation

---

## Conclusion

This experiment demonstrates the implementation of **YOLO for object detection**, highlighting its ability to detect multiple objects efficiently within an image.

YOLO’s real-time detection capability makes it one of the most widely used models in modern computer vision applications.

---

## Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning
