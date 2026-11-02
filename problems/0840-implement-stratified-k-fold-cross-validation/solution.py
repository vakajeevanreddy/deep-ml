import numpy as np

def stratified_kfold_indices(y, n_splits):
    y = np.asarray(y)
    
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
        
    classes, counts = np.unique(y, return_counts=True)
    
    if n_splits > counts.min():
        raise ValueError("n_splits cannot exceed the size of the smallest class")
        
    fold_test_indices = [[] for _ in range(n_splits)]
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        splits = np.array_split(cls_indices, n_splits)
        for fold_idx in range(n_splits):
            fold_test_indices[fold_idx].extend(splits[fold_idx])
            
    folds = []
    all_indices = np.arange(len(y))
    
    for fold_idx in range(n_splits):
        test_idx = np.array(sorted(fold_test_indices[fold_idx]))
        train_idx = np.setdiff1d(all_indices, test_idx)
        folds.append([train_idx.tolist(), list(test_idx)])
        
    return folds