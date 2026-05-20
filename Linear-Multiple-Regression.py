"""
Problem Statement:
Implement Simple and Multiple Linear Regression to predict continuous variables on the California Housing dataset. 
Perform data preprocessing (handle missing values, feature scaling). Evaluate models using MSE, RMSE, and R2 Score. 
Visualize the regression line and predictions.

Technical Breakdown & Line Mapping for Examiner:
- Lines 18-24 (Imports): Loads pandas, numpy, visualization libraries, dataset, preprocessing modules (Imputer, Scaler), and modeling/metric functions.
- Lines 26-28 (Data Ingestion): Fetches the California Housing data. X_full stores the feature matrix; y_full stores the continuous target values.
- Lines 30-31 (Imputation): SimpleImputer scans for NaN values and replaces them with the mathematical mean of their respective columns.
- Lines 33-35 (Scaling): StandardScaler normalizes the imputed features to have a mean of 0 and a standard deviation of 1.
- Line 37 (Data Splitting): train_test_split divides the dataset into 80% training data and 20% unseen testing data.
- Lines 39-45 (Simple Linear Regression): Isolates 'MedInc' as a 1D feature vector. LinearRegression() is initialized, fit via Ordinary Least Squares on the training data, and predicts on the test set. Metrics are calculated.
- Lines 47-53 (Multiple Linear Regression): Uses the entire n-dimensional scaled feature matrix. Fits the model, predicts, and calculates the same metrics.
- Lines 55-56 (Output): Prints the comparative mathematical metrics to the terminal.
- Lines 58-71 (Visualization): Generates a 1x2 subplot. The left plot plots the 2D simple regression line mapping MedInc to Price. The right plot compares Actual vs Predicted values for the Multiple regression model against an ideal 1:1 parity line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing()
X_full = pd.DataFrame(california.data, columns=california.feature_names)
y_full = california.target

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_full)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=california.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_full, test_size=0.2, random_state=42)

X_train_simple = X_train[['MedInc']]
X_test_simple = X_test[['MedInc']]
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"Simple LR  - MSE: {mse_simple:.4f} | RMSE: {rmse_simple:.4f} | R2: {r2_simple:.4f}")
print(f"Multi LR   - MSE: {mse_multi:.4f} | RMSE: {rmse_multi:.4f} | R2: {r2_multi:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test_simple, y_test, alpha=0.5, color='blue')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2)
plt.xlabel("Median Income (Scaled)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multi, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Linear Regression (Actual vs Predicted)")

plt.tight_layout()
plt.show()

"""
1. Terminal Metrics Evaluation

The terminal output provides a mathematical comparison of the two models' 
performance using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), 
and the Coefficient of Determination (R^2).

Simple Linear Regression: 
* MSE = 0.7091
* RMSE = 0.8421
* R^2 = 0.4589
Meaning: By using only a single feature (MedInc), the model can explain 
45.89% of the variance in housing prices.

Multiple Linear Regression: 
* MSE = 0.5559
* RMSE = 0.7456
* R^2 = 0.5758
Meaning: By incorporating all available features (e.g., house age, rooms, 
population, latitude/longitude), the error decreases and the explained 
variance increases to 57.58%.

Conclusion for Examiner: The Multiple Linear Regression model mathematically 
outperforms the Simple Linear Regression model because providing a 
multi-dimensional feature space reduces the residual error and captures 
a higher percentage of the data's variance.


2. Visual Plot Analysis

Left Plot: Simple Linear Regression
Axes: The X-axis represents the single scaled feature (MedInc), and the 
Y-axis represents the target (House Price).
Data Points (Blue): These are the actual recorded instances mapping income 
to house price.
Regression Line (Red): This represents the learned hypothesis function. It 
indicates a clear positive linear correlation: as median income increases, 
the predicted house price increases.
Data Artifact: Note the dense horizontal line of blue dots at exactly 5.0 
on the Y-axis. This represents an artificial cap in the dataset where all 
houses valued over 500,000 were logged as exactly 5.0.

Right Plot: Multiple Linear Regression
Axes: The X-axis represents the Actual Prices from the test set, while the 
Y-axis represents the model's Predicted Prices.
Ideal Line (Red): This is the parity line (y = x). If the model had an 
R^2 score of 1.0 (perfect accuracy), every single green dot would land 
exactly on this red line.
Data Points (Green): The spread of these dots around the red line visualizes 
the residual errors. The tighter the cluster around the red line, the better 
the predictions.
Data Artifact: You can see a strict vertical stack of green dots at exactly 
5.0 on the X-axis. Because the dataset artificially capped prices at 5.0, 
the linear model struggles to predict these values accurately, calculating 
them to be anywhere from 1.0 to over 7.0.
"""

"""
Assignment 2: Linear Regression
Concept: Supervised Parametric Regression
-----------------------------------------------------------------------------
Linear regression models the linear relationship between a scalar continuous 
dependent variable 'y' and one or more independent variables (features) 'X'.

* The Hypothesis Function: The model assumes the target can be calculated 
  by a linear combination of the input features plus a bias term:
  y = (w1 * x1) + (w2 * x2) + ... + (wn * xn) + bias
* Parameter Optimization (Ordinary Least Squares): The objective of the 
  algorithm is to find the optimal weights (w) and bias. It does 
  this by minimizing a cost function, typically the Mean Squared Error (MSE). 
  MSE Formula: MSE = Average of (Actual_y - Predicted_y)^2
* Simple vs. Multiple: Simple Linear Regression utilizes exactly a 
  1-dimensional feature vector. Multiple Linear Regression utilizes an 
  n-dimensional feature vector space.
  """