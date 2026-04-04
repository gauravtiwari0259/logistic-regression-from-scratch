# logistic-regression-from-scratch
Logistic Regression implemented from scratch (NumPy/Pandas) featuring Vectorized Gradient Descent, L2 Regularization, and Automated Alpha Tuning; validated against Scikit-Learn on the Pima Diabetes dataset.
# Logistic Regression from Scratch

This project implements **logistic regression from scratch using NumPy**, trained on the Pima Indians Diabetes dataset.
The goal is to understand the mathematical foundations of logistic regression by building it without relying on machine learning libraries.

The implementation is validated by comparing its performance with **scikit-learn**.

---

## Features

* Logistic regression implemented from scratch (NumPy)
* Vectorized gradient descent (no loops for training)
* Binary cross-entropy loss function
* Feature scaling (standardization)
* Learning rate (α) tuning using convergence plots
* L2 regularization (λ) to control overfitting
* Train-test split evaluation
* Comparison with scikit-learn implementation

---

## Dataset

* **Pima Indians Diabetes Dataset**
* Binary classification:

  * `1` → diabetic
  * `0` → non-diabetic

---

## Model Overview

The model predicts probability using the sigmoid function:

p = 1 / (1 + exp(-(X.w + b)))

### Training Details

## Feature Scaling

All features are standardized:

X = (X - μ) / σ

Training statistics are reused for test data to ensure consistency.

### Learning Rate (α) Tuning

Multiple values of α were tested to analyze convergence behavior.

* α = 0.1 → fast and stable convergence
* Smaller values → slower learning

---

### Regularization (λ)

L2 regularization was added to prevent overfitting:

J = loss + (λ / (2m)) * sum(w^2)

Observation:

* Increasing λ shrinks weights
* Large λ leads to underfitting

---

## Results

The dataset was split into:

* 80% training
* 20% testing

**Custom Model Accuracy:** 78.5714%%
**Scikit-learn Accuracy:** 76.6234%%

The close match confirms correctness of the implementation.

---

## Key Insights

* Feature scaling is essential for stable gradient descent
* Proper choice of α (alpha) significantly affects convergence speed
* Regularization reduces model complexity by shrinking weights
* Glucose emerged as the most influential feature based on learned weights

---

## Project Structure

```
.
├── diabetes_dataset.csv
├── diabetes_train.csv
├── diabetes_test.csv
├── logistic_regression_scratch.ipynb
├── alpha_tuning.png
├── lambda_tuning.png
└── README.md
```

---

## What I Learned

* How logistic regression works at a mathematical level
* How gradient descent updates parameters
* The importance of hyperparameter tuning
* The effect of regularization on model behavior
* How to validate a custom implementation using a standard library

---

## Future Improvements

* Wrap the NumPy engine in a FastAPI backend and build a Streamlit dashboard for real-time inference
* Benchmark results against XGBoost and Random Forests to capture complex, non-linear feature interaction
* Prioritize Recall (Sensitivity) and ROC-AUC over accuracy to minimize critical False Negatives in medical diagnosis

---

## Conclusion

This project demonstrates a complete implementation of logistic regression from scratch, along with proper evaluation and validation against a standard machine learning library.

It focuses on understanding the model beyond using pre-built tools.
