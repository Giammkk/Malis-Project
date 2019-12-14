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

def plotCatFeatures(x,y,name_features):
    plt.rcParams['figure.figsize'] = [9.5, 6]
    
    plt.figure(f)
    for f in range(len(x[0, :])):
        plt.scatter(x[:,f], y)
        plt.xlabel(name_features[f])
        plt.ylabel('PM2.5')
        plt.show()