import numpy as np
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = x - mean
    x = x / std
    
    return x, mean, std


def plotFeatures(x, y, name_features):
    #scatter_matrix(train, figsize=(30,20))
    plt.rcParams['figure.figsize'] = [9.5, 6]
    
    for f in range(len(x[0, :])):
        plt.figure(f)
        plt.scatter(x[:,f], y)
        plt.xlabel(name_features[f])
        plt.ylabel('PM2.5')
        plt.show()


def plotCatFeatures(x, y, name_features):
    color = ['b', 'r', 'g', 'y', 'c']
    plt.rcParams['figure.figsize'] = [9.5, 6]
    
    plt.figure()
    for f in range(len(name_features)):
        xx = x[:,f]
        plt.scatter(xx[xx==1]*f, y[xx==1], c=color[f])
        
    plt.xlabel(name_features)
    plt.ylabel('PM2.5')
    plt.show()
    
    
def classify(n):    
    if n <= 35:
        return 0
    if n <= 150:
        return 1
    else:
        return 2