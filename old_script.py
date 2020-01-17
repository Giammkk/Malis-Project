import sys
import pandas as pd
import numpy as np
import convert_data as cd
from minimization import LLS, conjugateGrad
import utils as ut
from sklearn.linear_model import LogisticRegression

np.random.seed(1)

#%% data preparation

data = pd.read_csv("Beijing_data.csv")
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

x = np.concatenate((months, hours, cbwd, x), axis=1)
x = np.delete(x, [12], axis=1)

np.random.shuffle(x)

# divide the dataset in training (80%) and test (20%)
xtrain = x[:143, :]#[: int(0.8 * tot_data), :]
xtest = x[144:311, :]#[int(0.8 * tot_data + 1) :, :]

ytrain_ = y[:143]#[: int(0.8 * tot_data), :]
ytest_ = y[144:311]#[int(0.8 * tot_data + 1) :, :]

ytrain = {}
ytrain['notnormalized'] = ytrain_.copy()
ytrain['normalized'], meany, stdy = ut.normalize(ytrain_)

ytest = {}
ytest['notnormalized'] = ytest_.copy()
ytest['normalized'], meany_test, stdy_test = ut.normalize(ytest_)

#%% Invert Iws (16), Is (17), Ir (18)

# for i in [16,17,18]:
#     minimum = np.min(xtrain[:,i])
#     maximum = np.max(xtrain[:,i])
#     nZeros = np.count_nonzero(xtrain[:,i] == 0)
    
#     print('min', i, minimum)
#     print('max', i, maximum)
#     print('num of zeros', i, nZeros)
#     print('-'*5)

# xtrain[:,16] = np.log(xtrain[:,16]) #Iws
# xtrain[:,17] = np.log(xtrain[:,17]) #Is
# xtrain[:,18] = np.log(xtrain[:,18]) #Ir

# for i in [16,17,18]:
#     minimum = np.min(xtrain[:,i])
#     maximum = np.max(xtrain[:,i])
#     nZeros = np.count_nonzero(xtrain[:,i] == 0)
    
#     print('min', i, minimum)
#     print('max', i, maximum)
#     print('n° zeros', i, nZeros)
#     print('-'*5)

#%% Square DEWP (13), TEMP (14), PRES (15)
    
# xtrain[:,13] = np.power(xtrain[:,13], 2)
# xtrain[:,14] = np.power(xtrain[:,14], 2)
# xtrain[:,15] = np.power(xtrain[:,15], 2)
    
# xtrain[:,13] = np.cbrt(xtrain[:,13])
# xtrain[:,14] = np.cbrt(xtrain[:,14])
# xtrain[:,15] = np.cbrt(xtrain[:,15])
    
# x13 = np.expand_dims( np.power(xtrain[:,13], 2), axis=1)
# x14 = np.expand_dims( np.power(xtrain[:,14], 2), axis=1)
# x15 = np.expand_dims( np.power(xtrain[:,15], 2), axis=1)

#%%
# xtrain = np.concatenate((x13, x14, x15, xtrain[:,13:16], xtrain[:, 16:19]), axis=1)
# xtrain = np.concatenate((xtrain, x13, x14, x15), axis=1)

#%% Normalize data
xtrain, meanx, stdx = ut.normalize(xtrain)
xtest, _, _ = ut.normalize(xtest)

#%% LLS
lls = LLS(ytrain['normalized'], xtrain)
lls.run()
lls.plot_w('LLS')

yhatLLS_train = lls.estimate(meany, stdy) 

errorLLS_train = lls.computeError(yhatLLS_train, ytrain['notnormalized'])
lls.ploty(yhatLLS_train, ytrain['notnormalized'])

acc_lls = lls.accuracy(yhatLLS_train, ytrain['notnormalized'])

#%% Conjugate Gradient 
cg = conjugateGrad(ytrain['normalized'], xtrain)
cg.run()
cg.plot_w('Conjugate Gradient')

yhatCG_train = cg.estimate(meany, stdy)
errorCG_train = cg.computeError(yhatCG_train, ytrain['notnormalized'])
cg.ploty(yhatCG_train, ytrain['notnormalized'])

acc_cg = cg.accuracy(yhatCG_train, ytrain['notnormalized'])

yhatCG_test = cg.test(ytest['normalized'], xtest, meany_test, stdy_test)
errorCG_test = cg.computeError(yhatCG_test, ytest['notnormalized'])
cg.ploty(yhatCG_test, ytest['notnormalized'])

acc_cg_test = cg.accuracy(yhatCG_test, ytest['notnormalized'])

#%% Logistic Regression

for i in range(len(ytrain['notnormalized'])):
    ytrain['notnormalized'][i] = ut.classify(ytrain['notnormalized'][i])
 
labels = ytrain['notnormalized'].astype(int).ravel()
logreg = LogisticRegression(max_iter=10e3, multi_class='multinomial', solver='lbfgs').fit(xtrain, labels)
score_train = logreg.score(xtrain, labels)*100
#-----------------------------------------------------------------------------
for i in range(len(ytest['notnormalized'])):
    ytest['notnormalized'][i] = ut.classify(ytest['notnormalized'][i])
    
labels = ytest['notnormalized'].astype(int).ravel()
score_test = logreg.score(xtest, labels)*100