# Lab 3: Effect of L1, L2, and Elastic Net Regularization on Model Training

## Overview

This experiment demonstrates the implementation and comparison of different regularization techniques used in machine learning models. Regularization helps prevent overfitting by adding a penalty term to the loss function.

In this lab, we implement and compare:

* **Linear Regression**
* **Ridge Regression (L2 Regularization)**
* **Lasso Regression (L1 Regularization)**
* **Elastic Net Regression (Combination of L1 and L2)**

The models are trained and evaluated using performance metrics to observe how regularization affects model training and prediction.

---

## Objectives

* To understand the concept of **regularization in machine learning**
* To implement **Ridge, Lasso, and Elastic Net regression models**
* To analyze the impact of regularization on model performance
* To evaluate models using standard performance metrics

---

## Libraries Used

The following Python libraries are used in this experiment:

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Dataset

The dataset used in this experiment contains both **numerical and categorical features**.

Data preprocessing steps include:

* Handling categorical variables using **One Hot Encoding**
* Feature scaling using **StandardScaler**
* Splitting dataset into **training and testing sets**

---

## Methodology

### 1. Data Preprocessing

* Import dataset using Pandas
* Separate features and target variable
* Apply preprocessing using **ColumnTransformer**
* Split dataset into training and testing sets

### 2. Model Implementation

The following regression models are implemented:

* **Linear Regression**
* **Ridge Regression (L2 Regularization)**
* **Lasso Regression (L1 Regularization)**
* **Elastic Net Regression (Combination of L1 and L2)**

### 3. Model Training

Each model is trained using the training dataset.

### 4. Model Evaluation

Models are evaluated using:

* **Mean Squared Error (MSE)**
* **R² Score**

These metrics help measure prediction accuracy and model performance.

---

## Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### R² Score

Represents how well the model explains the variance in the data.

---

## Observations

* **Ridge Regression** helps reduce model complexity and prevents overfitting by shrinking coefficients.
* **Lasso Regression** can shrink some coefficients to zero, effectively performing feature selection.
* **Elastic Net** combines both L1 and L2 penalties, balancing feature selection and coefficient shrinkage.

---

## Conclusion

Regularization techniques are essential for improving model generalization and preventing overfitting. In this experiment, Ridge, Lasso, and Elastic Net regression models were implemented and compared to analyze their impact on model performance.

Elastic Net provides a balance between Ridge and Lasso, making it useful when dealing with datasets containing multiple correlated features.

---

## Author

Samiksha Batra
MSc Artificial Intelligence & Machine Learning


