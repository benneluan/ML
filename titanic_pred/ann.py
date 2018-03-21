#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:34:15 2018

@author: ben
"""

import numpy as np
import pandas as pd

#Load the dataset
train_data = pd.read_csv('train.csv')

train_data = train_data.drop(["PassengerId","Name","Embarked","Ticket", "Fare", "Cabin"], axis = 1)

train_data.fillna(0, inplace = True)

X = train_data.iloc[:, 0:].values
y = train_data.iloc[:, 0].values
X = X[:, 1:]

#Encoding categorical data

from sklearn.preprocessing import  LabelEncoder, OneHotEncoder

labelencoder_X_1= LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X=X[:, 1:]

#Split the dataset into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y ,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()
#Adding the first hidden and the input  layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim=5))
#Adding the second hidden layer
classifier.add(Dense(units=6 , kernel_initializer ="uniform", activation = "relu"))
#Adding the output layer
classifier.add(Dense(units =1, kernel_initializer = "uniform", activation = "sigmoid"))
#compliing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#fitting the train set into the claissifier
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#initializing the given test datasets for the prediction
test_data = pd.read_csv("test.csv", sep=",")

test_data = test_data.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket", "Fare"], axis=1)

test_data.fillna(0, inplace = True)

z = test_data.iloc[: , 0:].values

labelencoder_z_1= LabelEncoder()

z[:, 1] = labelencoder_z_1.fit_transform(z[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])

z= onehotencoder.fit_transform(z).toarray()

z = z[:,1:]
 
new_prediction = classifier.predict(sc.transform(z))

new_prediction= (new_prediction > 0.5)

#Evaluating / Tunning the ANN to yield a better accuracy 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [20, 35],
              'epochs': [25, 250], 
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
#to get best predicttions 
best_parameters = grid_search.best_params_
#to get best accuracy of prediction
best_accuracy = grid_search.best_score_

new_pred = grid_search.predict(sc.transform(z))
new_pred = (new_pred  >  0.5)




