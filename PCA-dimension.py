"""
Problem Statement:
Load the Iris dataset, perform Principal Component Analysis (PCA) to reduce its dimensionality, 
visualize the PCA results by plotting the data points colored by species, and summarize how well 
the species are separated in the reduced-dimensional space.

Line-by-Line Execution Mapping
Lines 11-13: import pandas as pd, seaborn, matplotlib — Loads the libraries required for building dataframes (tables) and rendering the final 2D scatter plot.

Lines 14-16: from sklearn... — Imports the specific machine learning functions required for this assignment: the Iris dataset, the standardization math, and the PCA algorithm.

Line 18: iris_data = load_iris() — Loads the raw 4-dimensional dataset into memory.

Line 19: X = pd.DataFrame(...) — Extracts only the numerical feature data (sepal/petal lengths and widths) and structures it into a 2D table format.

Line 20: y = pd.Categorical.from_codes(...) — Extracts the target species (0, 1, 2) and converts them into human-readable text labels (setosa, versicolor, virginica) so the graph has proper names.

Line 22: scaler = StandardScaler() — Initializes the scaling algorithm.

Line 23: X_scaled = scaler.fit_transform(X) — Executes the mathematical normalization, forcing all data points to have a mean of 0 and standard deviation of 1.

Line 25: pca_model = PCA(n_components=2) — Initializes the Principal Component Analysis model and strictly limits the output to 2 dimensions.

Line 26: X_reduced = pca_model.fit_transform(X_scaled) — The core operation. It calculates the eigenvectors of the scaled data and compresses the 4 columns down into 2 principal components.

Line 28: final_df = pd.DataFrame(...) — Creates a brand new table specifically to hold the new 2-dimensional PC1 and PC2 data.

Line 29: final_df['Species'] = y — Re-attaches the text labels (setosa, etc.) to this new reduced table so the plotting library knows how to color-code the dots.

Line 31: plt.figure(figsize=(8, 6)) — Dictates the physical size of the pop-up window for the graph.

Line 32: sns.scatterplot(...) — Instructs Seaborn to map PC1 to the X-axis, PC2 to the Y-axis, and color the dots based on the Species column.

Lines 33-36: plt.xlabel, ylabel, title, grid — Adds the text labels and the background grid lines to make the plot readable.

Line 37: plt.show() — The execution command that forces Windows to draw and open the GUI window containing the finalized graph.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris_data = load_iris()
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_model = PCA(n_components=2)
X_reduced = pca_model.fit_transform(X_scaled)

final_df = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2'])
final_df['Species'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=final_df, palette='viridis', s=100, alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-Component PCA of Iris Dataset')
plt.grid(True)
plt.show()

"""
Visual Plot Analysis: 2-Component PCA of Iris Dataset

Axes: 
The X-axis represents Principal Component 1 (PC1), which is the mathematical 
direction in the original 4-dimensional feature space that captures the 
maximum amount of variance. The Y-axis represents Principal Component 2 (PC2), 
which captures the second highest amount of variance orthogonal to PC1.

Data Points: 
Each dot represents one of the 150 Iris flowers, mathematically compressed 
from its original 4 measurements (sepal/petal length and width) down onto 
this 2-dimensional Cartesian plane.

Cluster Separation & Conclusion for Examiner:
1. Setosa Cluster: Forms a completely distinct, isolated cluster on the 
left side of the plot. This proves that the underlying morphological features 
of Setosa are mathematically distinct from the other two species, making it 
trivially easy for an algorithm to classify.

2. Versicolor and Virginica Clusters: These form adjacent clusters on the 
right side of the plot with a slight boundary overlap. This indicates that 
these two species have highly correlated feature measurements. They share 
a very similar structural variance, making them slightly harder to isolate 
perfectly in a reduced dimensional space.

Summary Statement: 
The PCA successfully reduced the dimensionality of the dataset by 50% 
(from 4D to 2D) while preserving enough critical variance to clearly 
differentiate the species classes.
"""