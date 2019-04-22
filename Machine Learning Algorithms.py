#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:12:47 2019

@author: rafay
"""

import os
print(os.getcwd())
os.chdir("/Users/rafay/Pywork/justPractice") #Replace working directory as your own


################################
#### DATA
################################
warnings.filterwarnings('ignore')  # turn off all warnings

import requests
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
# download data 
r = requests.get("http://home.hampshire.edu/~emmCS/classes/machine_learning_2019/neural.npy", "neural.npy")
with open("neural.npy",'wb') as f: 
    f.write(r.content)


neural = np.load('neural.npy').item()

# split the data into training and test sets
y = neural['target']
X = neural['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



################################
#### SVM Q1
################################

#pt 1
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 10, gamma = 0.1)
classifier.fit(X_train, y_train)

print("SVC Classifier Score: ", classifier.score(X_test,y_test))

#pt 2
from sklearn.svm import LinearSVC
classifier_linear = LinearSVC()
classifier_linear.fit(X_train, y_train)
print("Linear SVC Score: ", classifier_linear.score(X_test,y_test))

################################
#### SVM 1b
################################
from sklearn.model_selection import GridSearchCV


# do grid search...
param_vals = np.power(10, np.arange(-3, 5, dtype = 'float64'))
param_grid = {'C': param_vals, 'gamma': param_vals}


gs = GridSearchCV(classifier,param_grid, cv=5)
gs.fit(X_train, y_train)

print("Grid Search Score: ", gs.score(X_test,y_test))

###

# # plot the grid search results
import pandas as pd
#
results = pd.DataFrame(gs.cv_results_)
scores = np.array(results.mean_test_score).reshape(param_vals.shape[0], param_vals.shape[0])

import matplotlib.pyplot as plt
plt.imshow(scores); 
plt.colorbar()



#BEst svc
svc = SVC(kernel = 'rbf', C = 10, gamma = 10).fit(X_train, y_train)
print("Best Params: ", (svc.score(X_test, y_test)))

################################
#### Normalization 1c
################################


# try normalizing the features...
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#linear again
from sklearn.svm import LinearSVC
classifier_linear = LinearSVC()
classifier_linear.fit(X_train, y_train)
print("Linear SVC Classifier Score with Normalization", classifier_linear.score(X_test,y_test))

# do grid search agauin
param_vals = np.power(10, np.arange(-3, 5, dtype = 'float64'))
param_grid = {'C': param_vals, 'gamma': param_vals}


gs = GridSearchCV(classifier,param_grid, cv=5)
gs.fit(X_train, y_train)

print("Grid Search Score with Normalization: ", gs.score(X_test,y_test))
print("Best params grid search: ", gs.best_params_)

svc = SVC(kernel = 'rbf', C = 1000, gamma = .001).fit(X_train, y_train)
print("Scaled RBF classifier score: ", (svc.score(X_test, y_test)))




###################
##################
##### Question 2
##################
##################


import numpy as np
import requests
from sklearn.model_selection import train_test_split


# download data 
r = requests.get("http://home.hampshire.edu/~emmCS/classes/machine_learning_2019/crime.npy", "crime.npy")
with open("crime.npy",'wb') as f: 
    f.write(r.content)


# load the data
crime = np.load('crime.npy').item()


# split the data into training and test sets
y = crime['target']
X = crime['data']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



###################
##################
##### Question 2 a
##################
##################


from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

#Regression SVM
regressor = SVR(kernel = 'rbf', C = 10, gamma = 0.1)
regressor.fit(X_train, y_train)

regressor.score(X_test,y_test)


#Linear SVM

regressor_linear = LinearSVR()
regressor_linear.fit(X_train, y_train)
regressor_linear.score(X_test,y_test)


# use a grid search to find the best parameters (see page 268)

###################
##################
##### Question 2 b
##################
##################
from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor(max_depth=4)
DT.fit(X_train,y_train)
DT.score(X_test,y_test)

#Visualize Decision tree
#Cant visualize graphviz here. DT on the worksheet



###################
##################
##### Question 2 c
##################
##################
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train,y_train)
print("Random Forest Score: ", regr.score(X_test,y_test))


feature_names = np.array(crime["feature_names"])
important_inds = (regr.feature_importances_ > 0)
importance_names = feature_names[important_inds]
importance_vals = regr.feature_importances_[important_inds]
plt.barh(range(len(importance_vals)), importance_vals);
plt.yticks(np.arange(len(importance_vals)), importance_names);


###################
##################
##### Question 2 d
##################
##################


from sklearn.ensemble import GradientBoostingRegressor

boost = GradientBoostingRegressor().fit(X_train, y_train)

print("Gradient boosting Score: ", (boost.score(X_test, y_test)))


###################
##################
##### Question 2 e
##################
##################

# compare ridge regression and the lasso - see my worksheet 6 exercise 2

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

# Ordinary least squares
ols = LinearRegression().fit(X_train, y_train)
print("OLS Score: ", (ols.score(X_test, y_test)))

ridge = Ridge(alpha = 10**-4).fit(X_train, y_train)
print("ridge regression score:", (ridge.score(X_test, y_test)))






