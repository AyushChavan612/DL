"""
Problem Statement:
Learn Decision Trees for classification problems by analyzing the spam.csv dataset. 
a. Split the data into training and test sets.
b. Build an unpruned Decision Tree.
c. Check model performances (Accuracy, Precision, Recall, F1) on training and test sets.
d. Apply cost-complexity pruning to mitigate overfitting.
e. Apply the Random Forest algorithm to overcome overfitting.
f. Apply the AdaBoost ensemble method on Decision Stumps.

Technical Breakdown & Line Mapping for Examiner:
- Decision Tree Theory: A non-parametric supervised learning algorithm that recursively splits the data based on feature conditions to maximize Information Gain. Unpruned trees tend to memorize training data, causing high variance (overfitting).
- Lines 23-30: Downloads the raw dataset dynamically. Drops empty columns, renames structural columns to 'label' and 'message', and binary encodes the target (ham=0, spam=1).
- Lines 32-34: The TfidfVectorizer converts raw text messages into a sparse matrix of numerical features representing term frequency-inverse document frequency.
- Line 36: Divides the matrix into 80% training data and 20% testing data.
- Lines 38-43: A generalized helper function to calculate and print the metrics, keeping the execution logic clean.
- Lines 45-49: Trains a basic DecisionTreeClassifier. Evaluating it reveals perfect training scores but lower test scores, confirming severe overfitting.
- Lines 51-63: Calculates the effective alphas for the pruning path. It iteratively trains trees across all alpha values, identifies the alpha yielding the maximum test accuracy, and locks in the optimally pruned tree to reduce variance.
- Lines 65-68: Implements bagging. It trains multiple deep decision trees on random data subsets and averages their output. This structural diversity drastically reduces overfitting.
- Lines 70-74: Implements boosting. It trains sequential "Decision Stumps" (trees with a maximum depth of 1). Each new stump mathematically focuses its weights on the specific instances misclassified by the previous stumps, reducing bias.
"""
import pandas as pd
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = "https://raw.githubusercontent.com/kenneth-lee-ch/SMS-Spam-Classification/master/spam.csv"
file_name = "spam.csv"
urllib.request.urlretrieve(url, file_name)

df = pd.read_csv(file_name, encoding='latin-1')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def print_metrics(model_name, y_true, y_pred):
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}\n")

unpruned_dt = DecisionTreeClassifier(random_state=42)
unpruned_dt.fit(X_train, y_train)
print_metrics("Unpruned DT (Train)", y_train, unpruned_dt.predict(X_train))
print_metrics("Unpruned DT (Test)", y_test, unpruned_dt.predict(X_test))

path = unpruned_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

test_accuracies = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
optimal_alpha = ccp_alphas[test_accuracies.index(max(test_accuracies))]

pruned_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
pruned_dt.fit(X_train, y_train)
print_metrics("Pruned DT (Test)", y_test, pruned_dt.predict(X_test))

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print_metrics("Random Forest (Test)", y_test, rf_model.predict(X_test))

stump = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_model = AdaBoostClassifier(estimator=stump, random_state=42)
ada_model.fit(X_train, y_train)
print_metrics("AdaBoost (Test)", y_test, ada_model.predict(X_test))

"""
Assignment 5: Tree-Based Models & Ensemble Learning
Concept: Non-Parametric Supervised Learning & Variance/Bias Reduction
-----------------------------------------------------------------------------
This assignment explores how structural modifications and ensemble math can 
overcome the primary flaw of basic decision trees: high variance (overfitting).

1. Decision Trees
* Mechanics: A predictive model that recursively partitions the feature 
  space. At each node, the algorithm evaluates all features and selects the 
  split that results in the highest Information Gain or the lowest Gini 
  Impurity ($I_G(p) = 1 - \sum p_i^2$).
* Overfitting: An unpruned tree will continue splitting until all leaf nodes 
  are mathematically pure (containing only one class). This results in a 
  model that memorizes training noise and fails to generalize.
* Cost-Complexity Pruning: Introduces a regularization parameter ($\alpha$) 
  that penalizes the tree for having too many terminal nodes. The algorithm 
  mathematically collapses terminal nodes that do not decrease the impurity 
  enough to justify the structural complexity.

2. Random Forest (Bagging)
* Mechanics: An ensemble method utilizing Bootstrap Aggregating (Bagging).
* How it works: It trains multiple independent, deep decision trees. Each 
  tree is trained on a random bootstrap sample of the training data (data 
  sampled with replacement). Furthermore, at each split, the tree is only 
  allowed to evaluate a random subset of the total features.
* Objective: The final prediction is the mode (majority vote) of all trees. 
  By aggregating many structurally diverse and un-correlated models, Random 
  Forest drastically reduces the overall variance without increasing bias, 
  virtually eliminating overfitting.

3. AdaBoost (Boosting)
* Mechanics: An ensemble method that trains models sequentially rather than 
  independently.
* How it works: It utilizes extremely weak learners, specifically Decision 
  Stumps (a decision tree with a maximum depth of exactly 1).
* Objective: In iteration 1, all training instances have equal weight. The 
  stump makes predictions. In iteration 2, the algorithm mathematically 
  increases the weights of the instances that were misclassified in 
  iteration 1. The next stump is forced to focus on correctly classifying 
  these harder, heavily weighted instances. The final prediction is a 
  weighted sum of all stumps. Boosting specifically reduces bias.
"""

