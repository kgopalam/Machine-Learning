#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
#    %matplotlib notebook
plt.figure()
plt.scatter(X_train, y_train, label='training data')
plt.scatter(X_test, y_test, label='test data')
plt.legend(loc=4);
	
	
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
plt.show()
def answer_one():
	res = np.zeros((4,100))
	
	y = np.sin(x)+x/6
	
	for i,degree in enumerate([1,3,6,9]):
		poly = PolynomialFeatures(degree)
		
		X_train_reshape = X_train[:,np.newaxis]
		X_test = np.linspace(0,10,100) 
		X_test_reshape = X_test[:,np.newaxis]
		
		X_train_poly = poly.fit_transform(X_train_reshape)
		X_test_poly = poly.fit_transform(X_test_reshape) 

		linreg = LinearRegression().fit(X_train_poly, y_train)
		y_predict = linreg.predict(X_test_poly)
		y_predict_f = y_predict.flatten()
		res[i,:]= y_predict_f 
	return res

def plot_one(degree_predictions):	
	plt.figure(figsize=(10,5))
	plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
	plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
	for i,degree in enumerate([1,3,6,9]):
		plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
	plt.ylim(-1,2.5)
	plt.legend(loc=4)
	plt.show()
	
plot_one(answer_one())

		
