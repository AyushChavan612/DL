"""
Problem Statement:
Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. 
Print both correct and wrong predictions. Python ML library classes can be used.

Technical Breakdown & Line Mapping for Examiner:
- KNN Theory: K-Nearest Neighbors is a non-parametric, instance-based learning algorithm. It classifies a new data point based on the majority class of its 'k' closest points in the feature space, typically calculated using Euclidean distance.
- Lines 18-21 (Imports): Loads the Iris dataset, data splitting module, KNN classifier, and accuracy metric from scikit-learn.
- Lines 23-26 (Data Ingestion): load_iris() retrieves the data. X stores the 4D feature matrix, y stores the numeric target labels, and target_names maps the numeric labels to string names.
- Line 28 (Data Splitting): train_test_split divides the data into 80% training instances and 20% unseen testing instances.
- Lines 30-31 (Model Training): Initializes KNeighborsClassifier with k=3. The fit() method executes the training phase, which in KNN simply stores the training data arrays in memory for distance computation later.
- Line 33 (Prediction): predict() computes the Euclidean distance between each test instance and all stored training instances, assigning the majority class label from the 3 nearest neighbors.
- Lines 35-49 (Evaluation & Output): Iterates through the actual and predicted labels simultaneously using zip(). Evaluates strict equality to determine 'Correct' or 'Wrong' status, accumulates counts, and prints a formatted terminal table.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris_data = load_iris()
X = iris_data.data
y = iris_data.target
target_names = iris_data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

print(f"{'Actual':<15} | {'Predicted':<15} | {'Status'}")
print("-" * 45)

correct_count = 0
wrong_count = 0

for actual_idx, pred_idx in zip(y_test, y_pred):
    actual_name = target_names[actual_idx]
    pred_name = target_names[pred_idx]
    
    if actual_idx == pred_idx:
        status = "Correct"
        correct_count += 1
    else:
        status = "WRONG"
        wrong_count += 1
        
    print(f"{actual_name:<15} | {pred_name:<15} | {status}")

print("-" * 45)
print(f"Total Correct: {correct_count}")
print(f"Total Wrong: {wrong_count}")

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

"""
Terminal Output Analysis: K-Nearest Neighbors Classification

Test Set Evaluation:
The terminal table displays the line-by-line evaluation of the 30 instances 
comprising the 20% testing split. Each row represents a single operational 
cycle where the algorithm computed the Euclidean distance between the unseen 
test instance and all 120 stored training instances.

Accuracy Metric (100%):
The algorithm successfully matched the predicted species to the actual 
species for all 30 test cases, resulting in 0 wrong predictions. A 100% 
accuracy rate is achievable on the Iris dataset using KNN because the 
feature space has low dimensionality (n=4), and the intra-class variance 
is small compared to the inter-class variance.

Conclusion for Examiner:
The k=3 hyperparameter successfully generalized the decision boundaries. 
By polling the 3 nearest neighbors rather than just 1, the model avoided 
overfitting to localized noise in the training data. This allowed it to 
correctly classify the mathematically overlapping regions between the 
Versicolor and Virginica clusters based purely on spatial proximity.
"""

"""
-----------------------------------------------------------------------------
Assignment 3: K-Nearest Neighbors (KNN)
Concept: Supervised Instance-Based (Lazy) Learning
-----------------------------------------------------------------------------
KNN is a non-parametric classification algorithm. It is considered "lazy" 
because it does not construct a generalized internal mathematical model 
during the training phase. Instead, it simply stores the entire training 
dataset in memory.

* Distance Metric: During the testing phase, the algorithm calculates the 
  distance between the unseen data point and all stored training points. 
  Euclidean Distance Formula: Distance = sqrt( (p1 - q1)^2 + (p2 - q2)^2 )
* Classification Rule: The algorithm identifies the 'k' training samples 
  that possess the smallest calculated distance to the test point. The test 
  point is then assigned the mathematical mode (majority vote) of those neighbors.
* The 'k' Hyperparameter: 
  - A smaller 'k' (e.g., k=1) creates complex, jagged decision boundaries 
    with low bias but high variance, making it highly susceptible to overfitting.
  - A larger 'k' mathematically smooths the decision boundaries, increasing 
    bias but lowering variance.
"""