# machine-learning
IRIS CLASSIFICATION

### Libraries Used:

1. **pandas (`import pandas as pd`):**
   - **Purpose:** Used for data manipulation and analysis.
   - **Working:** Reads and manipulates the dataset using Pandas DataFrames.
   - **Process:** `pd.read_csv("Iris.csv")` loads the Iris dataset into a Pandas DataFrame.
   - **Uses:** Efficient handling and exploration of tabular data.

2. **numpy (`import numpy as np`):**
   - **Purpose:** Provides support for large, multi-dimensional arrays and matrices.
   - **Working:** Handles numerical operations and array manipulations.
   - **Uses:** Efficient numerical computations and array handling.

3. **scikit-learn (`from sklearn.model_selection import train_test_split, from sklearn.linear_model import LogisticRegression, from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score`):**
   - **Purpose:** A machine learning library for classical algorithms.
   - **Working:** Splits data, builds logistic regression models, and evaluates model performance.
   - **Uses:** Standard machine learning algorithms for classification tasks.

4. **tensorflow (`import tensorflow as tf`):**
   - **Purpose:** Open-source machine learning framework for building and training neural networks.
   - **Working:** Defines, compiles, and trains a neural network model.
   - **Uses:** Deep learning tasks, including building and training neural networks.

5. **LabelEncoder (`from sklearn.preprocessing import LabelEncoder`):**
   - **Purpose:** Encodes categorical labels into numerical format.
   - **Working:** Converts string labels into numerical values.
   - **Uses:** Preprocessing categorical data for machine learning models.

6. **matplotlib (`import matplotlib.pyplot as plt`):**
   - **Purpose:** Visualization library for creating static, interactive, and animated plots.
   - **Working:** Plots ROC curves for model performance comparison.
   - **Uses:** Visualization of model evaluation metrics.

### Overall Working and Process:

1. **Dataset Loading and Exploration:**
   - The Iris dataset is loaded into a Pandas DataFrame (`df`) using `pd.read_csv`.
   - Basic information about the dataset, such as the first few rows and column names, is printed.

2. **Data Preprocessing:**
   - Features (`X`) and the target variable (`y`) are separated from the dataset.
   - If the target variable has string labels, a `LabelEncoder` is used to convert them into numerical format.
   - The data is split into training and testing sets using `train_test_split`.

3. **TensorFlow Model Building and Training:**
   - A simple neural network model is defined using TensorFlow's Keras API.
   - The model is compiled with an optimizer, loss function, and evaluation metric.
   - The model is trained using the training data.

4. **Model Evaluation and Prediction:**
   - The TensorFlow model is evaluated on the testing set, and accuracy is printed.
   - Class probabilities are obtained for each class using the trained TensorFlow model.

5. **Scikit-learn Model Building and Training:**
   - A logistic regression model is defined and trained using scikit-learn.
   - The scikit-learn model is evaluated on the testing set, and accuracy is printed.
   - Class probabilities are obtained for each class using the trained scikit-learn model.

6. **Model Comparison:**
   - ROC curves and AUC values are calculated for both TensorFlow and scikit-learn models.
   - F1 scores are calculated for both models.

7. **Visualization:**
   - ROC curves for both models are plotted using matplotlib.

### Uses and Applications:

- **Comparison of TensorFlow and Scikit-learn:**
  - The code provides a practical example of using both TensorFlow and scikit-learn for a classification task.
  - It demonstrates how to preprocess data, build, train, and evaluate models using two different machine learning frameworks.

- **Model Evaluation Metrics:**
  - The code calculates and compares accuracy, ROC AUC, and F1 score for both models.
  - This information is crucial for assessing the performance of machine learning models.

- **Visualization:**
  - The code includes the visualization of ROC curves, offering a graphical representation of model classification performance.

### How to Use:

1. **Dataset Path:**
   - Replace the dataset path in the `pd.read_csv` statement with the actual path to your Iris dataset.

2. **Library Installation:**
   - Ensure that the necessary libraries (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, and `matplotlib`) are installed. You can install them using `pip install library_name`.

3. **Run the Code:**
   - Execute the code in a Python environment. It will output model performance metrics and visualization.

