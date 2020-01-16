# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import convert_data as cd
from minimization import LLS, conjugateGrad
import utils as ut
from sklearn.linear_model import LogisticRegression

#%% data preparation

data = pd.read_csv("Beijing_data.csv")
# data = pd.read_csv("cities_data/ShanghaiPM20100101_20151231.csv")
# data = pd.read_csv("cities_data/ShenyangPM20100101_20151231.csv")
# data = pd.read_csv("cities_data/ChengduPM20100101_20151231.csv")
# data = pd.read_csv("cities_data/BeijingPM20100101_20151231.csv")
data = data.dropna()
statistics = data.describe()

name_features = data.columns.tolist()
name_features.pop(0)
name_features.remove('pm2.5')
# name_features.remove('PM_US Post')

head = data.head()
stats = data.describe()
tot_data = len(data)

y = data[['pm2.5']].values
# y = data[['PM_US Post']].values
y = np.array(y)

# plot data relations
#ut.plotFeatures(data[name_features].values, y, name_features)

x = data[['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']].values
# x = data[['DEWP', 'TEMP', 'PRES', 'HUMI', 'Iws', 'Iprec']].values
day = data[['day']].values
month = data[['month']].values
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

# x = np.concatenate((months, hours, cbwd, x), axis=1)
x = np.concatenate((hours, cbwd, x), axis=1)

#%% Prepare indexes to divide dataset in couples of weeks

from_ = 0
to_ = 0
r1 = 0
r2 = 0
fromTo = []
cnt = 0
for i in range(tot_data):
    cnt +=1
    from_, to_ = ut.fromTo(day.item(r1), month.item(r1))
    r2 = ut.findIndex(to_, day, month, r1)
    
    fromTo.append((r1,r2))
    
    if r2 == tot_data-1:
        break
    
    r1 = r2 + 1
    
#%% Prepare dataset, normalize data

accuracy_train_cg = []
accuracy_test_cg = []
accuracy_train_logreg = []
accuracy_test_logreg = []

for i in range(len(fromTo)-1):
    f0 = fromTo[i][0]
    t0 = fromTo[i][1]
    f1 = fromTo[i+1][0]
    t1 = fromTo[i+1][1]
    
    xtrain = x[f0:t0, :]
    xtest = x[f1:t1, :]
    
    ytrain_ = y[f0:t0]
    ytest_ = y[f1:t1]
    
    ytrain = {}
    ytrain['notnormalized'] = ytrain_.copy()
    ytrain['normalized'], meany, stdy = ut.normalize(ytrain_)
    
    ytest = {}
    ytest['notnormalized'] = ytest_.copy()
    ytest['normalized'], meany_test, stdy_test = ut.normalize(ytest_)

    xtrain, meanx, stdx = ut.normalize(xtrain)
    xtest, _, _ = ut.normalize(xtest)

    #%% LLS
    # lls = LLS(ytrain['normalized'], xtrain)
    # lls.run()
    # lls.plot_w('LLS')
    
    # yhatLLS_train = lls.estimate(meany, stdy) 
    
    # errorLLS_train = lls.computeError(yhatLLS_train, ytrain['notnormalized'])
    # lls.ploty(yhatLLS_train, ytrain['notnormalized'])
    
    # acc_lls = lls.accuracy(yhatLLS_train, ytrain['notnormalized'])

    #%% Conjugate Gradient 
    
    cg = conjugateGrad(ytrain['normalized'], xtrain)
    cg.run()
    # cg.plot_w('Conjugate Gradient')
    
    yhatCG_train = cg.estimate(meany, stdy)
    errorCG_train = cg.computeError(yhatCG_train, ytrain['notnormalized'])
    # cg.ploty(yhatCG_train, ytrain['notnormalized'])
    
    acc_cg = cg.accuracy(yhatCG_train, ytrain['notnormalized'])
    accuracy_train_cg.append(acc_cg)
    
    yhatCG_test = cg.test(ytest['normalized'], xtest, meany_test, stdy_test)
    errorCG_test = cg.computeError(yhatCG_test, ytest['notnormalized'])
    # cg.ploty(yhatCG_test, ytest['notnormalized'])
    
    acc_cg_test = cg.accuracy(yhatCG_test, ytest['notnormalized'])
    accuracy_test_cg.append(acc_cg_test)
    
    #%% Logistic Regression

    for i in range(len(ytrain['notnormalized'])):
        ytrain['notnormalized'][i] = ut.classify(ytrain['notnormalized'][i])
     
    labels = ytrain['notnormalized'].astype(int).ravel()
    logreg = LogisticRegression(max_iter=10e3, multi_class='multinomial', solver='lbfgs').fit(xtrain, labels)
    score_train = logreg.score(xtrain, labels)*100
    accuracy_train_logreg.append(score_train)
    #-------------------------------------------------------------------------
    for i in range(len(ytest['notnormalized'])):
        ytest['notnormalized'][i] = ut.classify(ytest['notnormalized'][i])
        
    labels = ytest['notnormalized'].astype(int).ravel()
    score_test = logreg.score(xtest, labels)*100
    accuracy_test_logreg.append(score_test)

#%% Statistics over accuracy
    
print('CG')
mean_train_cg = np.mean(np.array(accuracy_train_cg))
std_train_cg = np.std(np.array(accuracy_train_cg))
print("Mean accuracy train: %.2f" % mean_train_cg)
print("Std accuracy train: %.2f" % std_train_cg)
print("Max accuracy train: %.2f" % np.max(accuracy_train_cg))
print("Min accuracy train: %.2f" % np.min(accuracy_train_cg))

mean_test_cg = np.mean(np.array(accuracy_test_cg))
std_test_cg = np.std(np.array(accuracy_test_cg))
print("Mean accuracy test: %.2f" % mean_test_cg)
print("Std accuracy test: %.2f" % std_test_cg)
print("Max accuracy test: %.2f" % np.max(accuracy_test_cg))
print("Min accuracy test: %.2f" % np.min(accuracy_test_cg))

print('-'*50)

print('LogReg')
mean_train_logreg = np.mean(np.array(accuracy_train_logreg))
std_train_logreg = np.std(np.array(accuracy_train_logreg))
print("Mean accuracy train: %.2f" % mean_train_logreg)
print("Std accuracy train: %.2f" % std_train_logreg)
print("Max accuracy train: %.2f" % np.max(accuracy_train_logreg))
print("Min accuracy train: %.2f" % np.min(accuracy_train_logreg))

mean_test_logreg = np.mean(np.array(accuracy_test_logreg))
std_test_logreg = np.std(np.array(accuracy_test_logreg))
print("Mean accuracy test: %.2f" % mean_test_logreg)
print("Std accuracy test: %.2f" % std_test_logreg)
print("Max accuracy test: %.2f" % np.max(accuracy_test_logreg))
print("Min accuracy test: %.2f" % np.min(accuracy_test_logreg))

#%% Plot accuracies

ut.plotAccuracy('Conjugate Gradient', accuracy_train_cg, accuracy_test_cg)
ut.plotAccuracy('Logistic Regression', accuracy_train_logreg, accuracy_test_logreg)

