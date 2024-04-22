# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary .
4. Calculate the y-prediction.

## Program:

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRIYANKA S
RegisterNumber: 212222040125

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)



## Output:
![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/ee999063-7ed8-4f95-a617-73ba212944d7)

![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/acea08f8-1fe2-4a52-8b50-3137c67a3203)

![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/af0f114a-0c02-4fbd-bd99-cab435b62276)

![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/54718ac6-b4f4-4f7c-9cc0-6166a21c5d31)


![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/25f930a0-32f3-4629-878a-d16d0595306a)


![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/e2aa41d6-70da-44ce-a6a0-e391e20d9750)

![image](https://github.com/PriyaThilagam/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393798/2a49e724-24fc-4a2e-b660-e70dbf5f05b2)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

