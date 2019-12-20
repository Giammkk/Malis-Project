import sys
import pandas as pd
import numpy as np
import convert_data as cd
from minimization import *
import utils as ut
from sklearn.linear_model import LogisticRegression

# np.random.seed(1)

#%% data preparation

data = pd.read_csv("data.csv")
data = data.dropna()
statistics = data.describe()

name_features = data.columns.tolist()
name_features.pop(0)
name_features.remove('pm2.5')

head = data.head()
#print(data.dtypes)
stats = data.describe()
tot_data = len(data)

y = data[['pm2.5']].values
y = np.array(y)

# plot data relations
#ut.plotFeatures(data[name_features].values, y, name_features)

x = data[['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']].values
x = np.array(x)

# converting categorical features
months = data[['month']].values
months = cd.convertMonths(months, tot_data)
#ut.plotCatFeatures(months, y, ['winter', 'spring', 'summer', 'autumn'])

hours = data[['hour']].values
hours = cd.convertHours(hours, tot_data)
#ut.plotCatFeatures(hours, y, ['early morning', 'midday', 'afternoon', 'evening', 'night'])


cbwd = data[['cbwd']].values
cbwd = cd.convertCBWD(cbwd, tot_data)
#ut.plotCatFeatures(cbwd, y, ['cv', 'NW', 'NE', 'SE'])

x = np.concatenate((months, hours, cbwd, x), axis=1)

np.random.shuffle(x)

# divide the dataset in training (80%) and test (20%)
xtrain = x[: int(0.8 * tot_data), :]
xtest = x[int(0.8 * tot_data + 1) :, :]

ytrain_ = y[: int(0.8 * tot_data), :]
ytest = y[int(0.8 * tot_data + 1) :, :]

xtrain, meanx, stdx = ut.normalize(xtrain)

ytrain = {}
ytrain['notnormalized'] = ytrain_.copy()
ytrain['normalized'], meany, stdy = ut.normalize(ytrain_)

#%% Square DEWP (13), TEMP (14), PRES (15)
    
#xtrain[:,13] = np.power(xtrain[:,13], 2)
#xtrain[:,14] = np.power(xtrain[:,14], 2)
#xtrain[:,15] = np.power(xtrain[:,15], 2)
    
#xtrain[:,13] = np.cbrt(xtrain[:,13])
#xtrain[:,14] = np.cbrt(xtrain[:,14])
#xtrain[:,15] = np.cbrt(xtrain[:,15])
    
x13 = np.expand_dims( np.power(xtrain[:,13], 2), axis=1)
x14 = np.expand_dims( np.power(xtrain[:,14], 2), axis=1)
x15 = np.expand_dims( np.power(xtrain[:,15], 2), axis=1)

# xtrain = np.concatenate((x13, x14, x15, xtrain[:,13:16], xtrain[:, 16:19]), axis=1)
xtrain = np.concatenate((x13, x14, x15, xtrain), axis=1)


x13 = np.expand_dims( np.power(xtest[:,13], 2), axis=1)
x14 = np.expand_dims( np.power(xtest[:,14], 2), axis=1)
x15 = np.expand_dims( np.power(xtest[:,15], 2), axis=1)

xtest = np.concatenate((x13, x14, x15, xtest), axis=1)
#%% Logistic Regression

for i in range(len(ytrain['notnormalized'])):
    ytrain['notnormalized'][i] = ut.classify(ytrain['notnormalized'][i])
 
labels = ytrain['notnormalized'].astype(int).ravel()
logreg = LogisticRegression(max_iter=10e4, multi_class='multinomial', solver='lbfgs').fit(xtrain, labels)
# aaa = logreg.predict(xtrain)
score_train = logreg.score(xtrain, labels)

for i in range(len(ytest)):
    ytest[i] = ut.classify(ytest[i])
 
labels = ytest.astype(int).ravel()
score_test = logreg.score(xtest, labels)