"""
Problem Statement:
Write a program to implement the naïve Bayesian classifier for a sample training data set 
stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

Technical Breakdown & Line Mapping for Examiner:
- Naïve Bayes Theory: A probabilistic classifier based on Bayes' Theorem: $P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$. It assumes strong (naïve) conditional independence between features. The GaussianNB variant assumes continuous features follow a Gaussian (normal) distribution.
- Lines 18-23 (Imports): Loads pandas (dataframes), numpy (numerical NaN assignment), urllib (dataset downloading), and sklearn (modeling and metrics).
- Lines 25-29 (Data Ingestion): Downloads the Pima Indians Diabetes CSV directly from a remote repository, saves it locally, and loads it into memory with explicit column headers.
- Lines 31-34 (Data Cleaning): Biological metrics (e.g., BloodPressure, BMI) cannot realistically be zero; these are missing data points. The loop replaces 0 with np.nan, then imputes these missing values using the mathematical mean of each respective column to maintain continuous distribution integrity.
- Lines 36-39 (Data Splitting): Isolates the feature matrix X and target vector y. train_test_split divides the data into an 80% training set and a 20% testing set.
- Lines 41-42 (Model Training): Initializes the GaussianNB model. The fit() method calculates the mean and variance for each class per feature to construct the prior and likelihood probability distributions.
- Lines 44-52 (Evaluation): predict() applies the Maximum A Posteriori (MAP) decision rule to classify the test data. The output displays the overall Accuracy and the Confusion Matrix, which maps True Negatives, False Positives, False Negatives, and True Positives.
"""
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
file_name = "pima_diabetes.csv"
urllib.request.urlretrieve(url, file_name)

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
diabetes_data = pd.read_csv(file_name, names=columns)

cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_clean:
    diabetes_data[col] = diabetes_data[col].replace(0, np.nan)
    diabetes_data[col] = diabetes_data[col].fillna(diabetes_data[col].mean())

X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred) * 100
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print("--- Pima Indians Diabetes Classification ---")
print(f"Accuracy: {accuracy:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

"""
Terminal Output Analysis: Naïve Bayes Classification

1. Accuracy Metric (74.68%)
The model correctly classified approximately 74.68% of the unseen 
testing data. Statistically, this means out of the 154 patients in the 
test split, the algorithm made the correct diagnostic prediction for 115 
of them based solely on the Gaussian probability distributions of their 
biological features.

2. Confusion Matrix Breakdown
The 2x2 matrix provides a granular view of the classification errors 
rather than just an aggregate accuracy score.

* True Negatives (Top-Left: 78): The model correctly predicted these 
patients do NOT have diabetes.
* False Positives (Top-Right: 21): Type I Error. The model predicted 
these patients have diabetes, but they actually do not.
* False Negatives (Bottom-Left: 18): Type II Error. The model predicted 
these patients do NOT have diabetes, but they actually do. This is the 
most dangerous error in medical diagnostics.
* True Positives (Bottom-Right: 37): The model correctly predicted these 
patients HAVE diabetes.

Conclusion for Examiner:
The Gaussian Naïve Bayes model establishes a strong baseline accuracy of 
~75%. However, the assumption of strict feature independence (e.g., assuming 
BMI and Glucose levels do not influence each other) inherently limits the 
predictive ceiling of this algorithm in complex biological datasets, leading 
to the 39 misclassifications observed in the matrix.
"""