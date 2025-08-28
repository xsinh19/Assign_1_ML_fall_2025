    from typing import Union
    import pandas as pd
    
    
    def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
        """
        Function to calculate the accuracy
        """
    
        """
        The following assert checks if sizes of y_hat and y are equal.
        Students are required to add appropriate assert checks at places to
        ensure that the function does not fail in corner cases.
        """
        assert y_hat.size == y.size
        correct_pred = float(y_hat[y_hat==y].size)
        return float(correct_pred / y_hat.size)
    
    
    def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
        """
        Function to calculate the precision
        """
        assert y_hat.size == y.size
        correct_pred_cls = y_hat[(y_hat==cls) & (y==cls)].size
        return correct_pred_cls/ y_hat[y_hat==cls].size
        
    
    
    def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
        """
        Function to calculate the recall
        """
        assert y_hat.size == y.size
        correct_pred_cls = y_hat[(y_hat==cls) & (y==cls)].size
        return correct_pred_cls / y[y==cls].size
        
    
    
    def rmse(y_hat: pd.Series, y: pd.Series) -> float:
        """
        Function to calculate the root-mean-squared-error(rmse)
        """
        assert y_hat.size == y.size
        return (((y_hat - y)**2).mean())**0.5
    
    
    
    
    def mae(y_hat: pd.Series, y: pd.Series) -> float:
        """
        Function to calculate the mean-absolute-error(mae)
        """
    
        assert y_hat.size == y.size
        return (abs(y_hat - y)).mean()
    
