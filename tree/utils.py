"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np 
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data

    
    """
    encoded_X = pd.get_dummies(X, drop_first=False, dummy_na=False)
    return encoded_X

    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # check whether the input is discrete or continous if the input is continous then true and it the input is discrete the false
    
    return pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_integer_dtype(y)

    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # values give us the probability of the category 
    
    Entropy = 0 
    values = Y.value_counts(normalize=True)
    for p in values:
        if p>0:
            Entropy = Entropy - p*(np.log2(p))

    return Entropy
    
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    gini = 0
    value = Y.value_counts(normalize=True, dropna= False)
    for p in value:
        if p>0:
            gini = gini + (p*(1-p))
    return gini 

    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    if criterion == "entropy":
        original_impurity = entropy(Y)
    elif criterion == "gini" or criterion == "gini_index":
        original_impurity = gini_index(Y)
    elif criterion == "MSE":
        original_impurity = Y.var()
    else:
        raise ValueError("Invalid Criterion")

    weighted_impurity = 0

    for value, subset in Y.groupby(attr, observed=True):
        weight = len(subset) / len(Y)
        if criterion == "entropy":
            impurity = entropy(subset)
        elif criterion == "gini" or criterion == "gini_index":
            impurity = gini_index(subset)
        else:
            impurity = subset.var()
        weighted_impurity += weight * impurity

    return original_impurity - weighted_impurity


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_gain = -np.inf
    best_attr = None
    best_threshold = None

    for feature in features:
        if check_ifreal(X[feature]):
            sorted_values = np.sort(X[feature].unique())
            threshold = (sorted_values[:-1] + sorted_values[1:])/2

            for t in threshold:
                left_mask = X[feature] <= t
                gain = information_gain(y,left_mask, criterion)

                if gain > best_gain:
                    best_gain = gain
                    best_attr = feature
                    best_threshold = t

      
        else:
            gain = information_gain(y,X[feature],criterion)

            if gain > best_gain:
                best_gain = gain 
                best_attr = feature
                best_threshold = None
    
    
    return best_attr, best_threshold  
    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    if check_ifreal(X[attribute]):
        left_mask = X[attribute] <=value
        right_mask = X[attribute] > value

        X_left,y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return (X_left,y_left),(X_right, y_right)

    else:
        mask = X[attribute] == value
        X_subset, y_subset = X[mask],y[mask]

        return X_subset, y_subset
    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    

    pass
