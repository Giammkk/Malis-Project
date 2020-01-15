import numpy as np
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from calendar import monthrange

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1
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


def daysOfMonth(month):
    return monthrange(2010, month)[1]
    
def until(day, month):
    ndays = daysOfMonth(month)
    return (day + 7 - 1) % ndays


def fromTo(day, month):
    from_ = day
    to_ = until(day, month)
    
    return from_, to_

def findIndex(to, day, month, r1):
    flag = 0
    prev = day.item(r1)
    now = 0
    cnt = 0
    
    for i in range(r1, len(day)):
        now = day.item(i)
        
        if prev != now:
            prev = now
            cnt += 1
            
        if cnt == 7:
            flag = 1
            
        if flag == 1 and to != day.item(i):
            return i-1
        
    return len(day) - 1 
        

def plotAccuracy(algorithm, train, test):
    plt.figure()
    plt.xlabel('week')
    plt.ylabel('accuracy')
    plt.title('Classification rate for each week (' + algorithm + ')')
    plt.grid()
    plt.plot(range(len(train)), train, '.--r')
    plt.plot(range(len(test)), test, '.--b')
    plt.legend(['train', 'test'])
    plt.show()