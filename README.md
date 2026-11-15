# Iris Flower Classification using Machine Learning

## Project Overview
This project focuses on classifying Iris flowers into three different species using machine learning. The classification is performed using the **K-Nearest Neighbors (KNN)** algorithm with the Scikit-learn library.

The model uses flower measurements to predict whether a flower belongs to Iris-setosa, Iris-versicolor, or Iris-virginica.

## Objective
The main objective is to build and evaluate a machine learning classification model that can accurately predict Iris flower species based on their physical measurements.

## Dataset Description
The project uses the Iris dataset (`Iris.csv`), which contains 150 observations and the following features:

- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

**Target Variable:** Species

The dataset contains three flower species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Technologies and Libraries Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
1. Imported the required Python libraries.
2. Loaded and explored the Iris dataset.
3. Checked the dataset for missing values and examined the distribution of flower species.
4. Separated the input features and target variable.
5. Split the dataset into training and testing sets (80:20).
6. Applied feature scaling using StandardScaler.
7. Built and trained a K-Nearest Neighbors (KNN) classification model.
8. Predicted flower species using the test dataset.
9. Evaluated model performance using accuracy, confusion matrix, and classification report.

## Machine Learning Algorithm
**K-Nearest Neighbors (KNN)**

- Algorithm: K-Nearest Neighbors Classifier
- Number of Neighbors (K): 5
- Feature Scaling: StandardScaler
- Training Data: 80%
- Testing Data: 20%

## Model Performance
The model achieved an accuracy of **93.33%** on the test dataset.

### Confusion Matrix

| Actual / Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 10 | 0 |
| Virginica | 0 | 2 | 8 |

### Classification Report

| Species | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 0.83 | 1.00 | 0.91 |
| Iris-virginica | 1.00 | 0.80 | 0.89 |

## Key Findings
- The model achieved 93.33% test accuracy.
- Iris-setosa was classified correctly in all 10 test cases.
- Iris-versicolor was classified correctly in all 10 test cases.
- Eight out of ten Iris-virginica samples were classified correctly.
- Two Iris-virginica samples were predicted as Iris-versicolor.

## Conclusion
The project demonstrates how machine learning can be used to classify Iris flower species based on their measurements. The KNN model achieved 93.33% accuracy on the test dataset, showing good classification performance.

This project also provided practical experience in data preprocessing, feature scaling, model training, prediction, and model evaluation using Python and Scikit-learn.

## How to Run This Project

1. Clone or download this repository.
2. Make sure Python and Jupyter Notebook are installed.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `iris_flower_classification.ipynb`.
6. Run all notebook cells.

**Important:** Keep `Iris.csv` in the same folder as the notebook.

## Project Files
- `Iris_flower_classification.ipynb` – Complete project notebook.
- `Iris.csv` – Dataset used for classification.
- `README.md` – Project documentation.

---

**Project:** CodeAlpha Data Science Internship – Task 1  
**Author:** Aryan  
**GitHub:** [Aryan18-03](https://github.com/Aryan18-03)
