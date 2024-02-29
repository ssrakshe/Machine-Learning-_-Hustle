#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
import copy, math


# In[100]:


data = pd.read_csv('data12.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values


# In[101]:


X.shape,y.shape


# In[102]:


def sigmoid(z):
   
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g


# In[103]:


def compute_cost_logistic_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    epsilon = 1e-15  # Small value to avoid log(0) issues

    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        cost += -y[i] * np.log(f_wb_i + epsilon) - (1 - y[i]) * np.log(1 - f_wb_i + epsilon)

    regularization_term = (lambda_ / (2 * m)) * np.sum(w**2)
    cost = (1 / m) * cost + regularization_term

    return cost


# In[104]:


def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    m, n = X.shape
    dj_dw = np.zeros_like(w)
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        dj_dw += err_i * X[i].reshape(-1, 1)
        dj_db += err_i

    dj_dw /= m
    dj_db /= m

    dj_dw += (lambda_ / m) * w

    return dj_db, dj_dw


# In[105]:


def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
   
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    l=0.1
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b,l)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic_reg(X, y, w, b,l) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history  


# In[106]:


w_tmp = np.ones((X.shape[1], 1))
b_tmp = 0
alpha=0.01
lambda_tmp = 0.1
iteration=1000
w_out, b_out,_ =  gradient_descent(X, y, w_tmp, b_tmp,alpha, iteration)
print(f"\nupdated parameters: w:{w_out.flatten()}, b:{b_out}")

