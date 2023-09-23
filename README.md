# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b   importing     the required modules from sklearn.
6. Obtain the graph.
 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:  Yuvabharathi.B
RegisterNumber:  212222230181

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()
def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)
def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

# Output:
## Array value of X:
![21](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/88b00d24-7c4f-4bf7-9e06-e6a31f8d8e27)
## Array value of Y:
![22](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/2172c11d-fd2b-4c09-b2ae-eccc0d3e04c5)
## Exam 1-Score graph:
![23](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/581fdd00-c99b-4bde-900f-28b94253786f)
## Sigmoid function graph:
![24](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/f82dcffe-c7fe-4fd1-8514-e70f55b71808)
## X_Train_grad value:
![25](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/5411c983-839d-4f59-8c79-9fd6e889de18)
## Y_Train_grad value:
![26](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/7c38b0ad-09f2-43d2-9fe8-44e4ff45e165)
## Print res.X:
![27](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/7c1fdd2e-31f4-499c-bd0d-282935777823)
## Decision boundary-gragh for exam score:
![28](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/caf74e64-6d09-4ea1-b57c-038fd3e05b77)
## Probability value:
![29](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/ff5cfea4-0746-4b1d-932a-4c89b46f6484)
## Prediction value of mean:
![30](https://github.com/hariprasath5106/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/111515488/5891908a-41ac-495c-99d7-4d388232886a)

# Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
