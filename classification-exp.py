import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

#(A)

# 1. Import required libraries
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tree.base import DecisionTree
from metrics import accuracy, precision, recall  # assuming these are in metrics.py

X_df = pd.DataFrame(X,columns = ["Feature 1", "Feature 2"])
y_s = pd.Series(y)


# train and test splittingthe data 
X_train, X_test, y_train, y_test = train_test_split(X_df,y_s,test_size= 0.3, random_state = 42)

#Model fit 
tree = DecisionTree(criterion = "information gain", max_depth = 5)
tree.fit(X_train,y_train)

#predict
y_pred = tree.predict(X_test)

#Metrices
print("Accuracy:", accuracy(y_pred, y_test))

for cls in y_s.unique():
    print(f"Class {cls} - Precision:", precision(y_pred, y_test, cls))
    print(f"Class {cls} - Recall:", recall(y_pred, y_test, cls))


#(b)

# Define parameter range
depth_values = [1, 2, 3, 4, 5, 6]

# Outer 5-fold cross-validation
kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

for train_index, test_index in kf_outer.split(X_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_sr.iloc[train_index], y_sr.iloc[test_index]

    # Inner CV to select best depth
    best_depth = None
    best_score = -np.inf

    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    for d in depth_values:
        inner_scores = []
        for inner_train, inner_val in kf_inner.split(X_train):
            X_inner_train, X_inner_val = X_train.iloc[inner_train], X_train.iloc[inner_val]
            y_inner_train, y_inner_val = y_train.iloc[inner_train], y_train.iloc[inner_val]

            tree = DecisionTree(criterion="information_gain", max_depth=d)
            tree.fit(X_inner_train, y_inner_train)
            y_val_pred = tree.predict(X_inner_val)
            inner_scores.append(accuracy(y_val_pred, y_inner_val))

        avg_score = np.mean(inner_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_depth = d

    # Train on full training set using best depth
    final_tree = DecisionTree(criterion="information_gain", max_depth=best_depth)
    final_tree.fit(X_train, y_train)
    y_pred_outer = final_tree.predict(X_test)
    outer_scores.append(accuracy(y_pred_outer, y_test))

print("Cross-Validation Scores:", outer_scores)
print("Mean Accuracy:", np.mean(outer_scores))


best_depths_per_fold = []  # store best depth for each fold

for train_index, test_index in kf_outer.split(X_df):
    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_sr.iloc[train_index], y_sr.iloc[test_index]

    best_depth = None
    best_score = -np.inf

    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    for d in depth_values:
        inner_scores = []
        for inner_train, inner_val in kf_inner.split(X_train):
            X_inner_train, X_inner_val = X_train.iloc[inner_train], X_train.iloc[inner_val]
            y_inner_train, y_inner_val = y_train.iloc[inner_train], y_train.iloc[inner_val]

            tree = DecisionTree(criterion="information_gain", max_depth=d)
            tree.fit(X_inner_train, y_inner_train)
            y_val_pred = tree.predict(X_inner_val)
            inner_scores.append(accuracy(y_val_pred, y_inner_val))

        avg_score = np.mean(inner_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_depth = d

    best_depths_per_fold.append(best_depth)  # Save best depth for this outer fold

print("Best Depths per Fold:", best_depths_per_fold)
