# Logistic Regression from Scratch - Breast Cancer Classification

## Overview
This project implements logistic regression from scratch using Python and NumPy to classify breast cancer tumors as benign or malignant based on given features.

## Features
- Implements logistic regression without using external machine learning libraries.
- Uses gradient descent for parameter optimization.
- Supports data preprocessing, feature scaling, and model evaluation.
- Uses the Breast Cancer Wisconsin dataset for classification.


## Implementation Details
The logistic regression algorithm is implemented using:
- **Sigmoid Function**: Converts linear outputs into probabilities.
- **Cost Function**: Uses log loss to measure performance.
- **Gradient Descent**: Optimizes weights by minimizing the cost function.

## Example Visualization
![download](https://github.com/user-attachments/assets/fe3d828c-094d-4156-8d25-beed5d6fcde8)

The scatter plot represents two features from the Breast Cancer Wisconsin Dataset:

 - X-axis: mean radius (first feature in the dataset)
 - Y-axis: mean texture (second feature in the dataset)
These features describe tumor characteristics based on digitized images of fine needle aspirate (FNA) of breast masses.
The colors in the scatter plot represent the predicted class of each tumor:
 - Green : Predicted as benign (0) (non-cancerous).
 - Dark Blue: Predicted as malignant (1) (cancerous).
