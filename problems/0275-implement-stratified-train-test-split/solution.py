import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    train_indices = []
    test_indices = []
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
        
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    return (
        X[train_indices], X[test_indices],
        y[train_indices], y[test_indices]
    )