import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    X_train_arr = np.asarray(X_train,dtype = np.float64)
    X_test_arr = np.asarray(X_test,dtype = np.float64)
    mean = np.mean(X_train_arr,axis = 0)
    std = np.std(X_train_arr,axis = 0,ddof = 0)
    safe_std = np.where(std == 0.0,1.0,std)
    return(X_test_arr-mean)/safe_std