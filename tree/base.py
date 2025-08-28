from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)


# Node structure to represent tree nodes
@dataclass
class Node:
    attribute: str = None
    threshold: float = None
    children: dict = None
    prediction: object = None


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train and construct the decision tree."""
        self._is_regression = check_ifreal(y)
        self.root = self._build_tree(X, y, depth=0)

    def _criterion_for_utils(self, y: pd.Series) -> str:
        """Map criterion to correct string for utils functions."""
        if check_ifreal(y):
            return "MSE"
        if self.criterion == "information_gain":
            return "entropy"
        elif self.criterion == "gini_index":
            return "gini_index"
        return "entropy"

    def _leaf_value(self, y: pd.Series):
        """Return prediction for leaf node."""
        if check_ifreal(y):
            return float(y.mean())
        return y.mode(dropna=False).iloc[0]

    def _pure(self, y: pd.Series) -> bool:
        return y.nunique(dropna=False) <= 1

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        # Stopping conditions
        if len(X) == 0:
            return Node(prediction=None)
        if self._pure(y):
            return Node(prediction=y.iloc[0])
        if depth >= self.max_depth or X.shape[1] == 0:
            return Node(prediction=self._leaf_value(y))

        # Find best attribute and threshold
        features = X.columns
        util_criterion = self._criterion_for_utils(y)
        best_attr, best_threshold = opt_split_attribute(X, y, util_criterion, features)

        if best_attr is None:
            return Node(prediction=self._leaf_value(y))

        node = Node(attribute=best_attr, threshold=best_threshold)

        if best_threshold is not None:  # Continuous
            (X_left, y_left), (X_right, y_right) = split_data(X, y, best_attr, best_threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                return Node(prediction=self._leaf_value(y))
            node.children = {
                "left": self._build_tree(X_left, y_left, depth + 1),
                "right": self._build_tree(X_right, y_right, depth + 1),
            }
        else:  # Discrete
            node.children = {}
            for val in X[best_attr].unique():
                X_sub, y_sub = split_data(X, y, best_attr, val)
                if len(y_sub) == 0:
                    node.children[val] = Node(prediction=self._leaf_value(y))
                else:
                    X_child = X_sub.drop(columns=[best_attr], errors="ignore")
                    node.children[val] = self._build_tree(X_child, y_sub, depth + 1)

        node.prediction = self._leaf_value(y)
        return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict values for input data."""
        return X.apply(lambda row: self._predict_row(row, self.root), axis=1)

    def _predict_row(self, row: pd.Series, node: Node):
        if node.attribute is None or node.children is None:
            return node.prediction
        if node.threshold is not None:  # Continuous
            val = row.get(node.attribute, np.nan)
            branch = "left" if pd.notna(val) and val <= node.threshold else "right"
            child = node.children.get(branch)
            return self._predict_row(row, child) if child else node.prediction
        else:  # Discrete
            val = row.get(node.attribute, None)
            child = node.children.get(val)
            return self._predict_row(row, child) if child else node.prediction

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.root is None:
            print("(empty tree)")
            return
        self._print_tree(self.root, indent="")

    def _print_tree(self, node: Node, indent: str):
        if node.attribute is None or node.children is None:
            print(indent + f"Predict: {node.prediction}")
            return
        if node.threshold is not None:  # Continuous
            print(indent + f"? ({node.attribute} <= {node.threshold})")
            print(indent + "  Y:", end=" ")
            self._print_tree(node.children.get("left"), indent + "    ")
            print(indent + "  N:", end=" ")
            self._print_tree(node.children.get("right"), indent + "    ")
        else:  # Discrete
            print(indent + f"? ({node.attribute})")
            for val, child in node.children.items():
                print(indent + f"  {val}:", end=" ")
                self._print_tree(child, indent + "    ")
