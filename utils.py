import numpy as np

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = x - mean
    x = x / std
    
    return x, mean, std