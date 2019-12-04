import pandas as pd
import numpy as np
import convert_data as cd
from minimization import *
import utils as ut

#np.random.seed(1)

#%% data preparation

data = pd.read_csv("data.csv")
data = data.dropna()

head = data.head()
#print(data.dtypes)
stats = data.describe()
tot_data = len(data)

x = data[['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']].values # QUESTION: should we consider the year?
x = np.array(x)
y = data[['pm2.5']].values
y = np.array(y)

# converting categorical features
months = data[['month']].values
months = cd.convertMonths(months, tot_data)

hours = data[['hour']].values
hours = cd.convertHours(hours, tot_data)

cbwd = data[['cbwd']].values
cbwd = cd.convertCBWD(cbwd, tot_data)

x = np.concatenate((months, hours, cbwd, x), axis=1)

np.random.shuffle(x)

# divide the dataset in training (80%) and test (20%)
xtrain = x[: int(0.8 * tot_data), :]
xtest = x[int(0.8 * tot_data + 1) :, :]

ytrain = y[: int(0.8 * tot_data), :]
ytest = y[int(0.8 * tot_data + 1) :, :]

xtrain, meanx, stdx = ut.normalize(xtrain)
ytrain_notnorm = ytrain.copy()
ytrain, meany, stdy = ut.normalize(ytrain)

#%% LLS
lls = LLS(ytrain, xtrain)
lls.run()
lls.plot_w()
yhat = lls.estimate(meany, stdy)
errorLLS = lls.computeError(yhat, ytrain_notnorm)

#CHECK NORMALIZATION 

#for i in range(xtrain.shape[1]):
#    sum = np.sum(xtrain[:, i])
#    std = np.std(xtrain[:, i])
#    print(i , sum, std)

#%% Conjugate Gradient 
cg = conjugateGrad(ytrain, xtrain)
cg.run()
cg.plot_w()
yhat = lls.estimate(meany, stdy)
errorCG = lls.computeError(yhat, ytrain_notnorm)
