#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('.\deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients


# In[3]:


x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])


# In[5]:


def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb - y[i])**2
        total_cost=1/(2*m)*cost
        
    return total_cost


# In[8]:


def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=f_wb-y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw=dj_dw / m 
    dj_db=dj_db / m
    
    return dj_dw, dj_db


# In[9]:


plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()


# In[12]:


def gradient_descent(x,y,w_in,b_in,alpha, num_iters,cost_function, gradient_function):
    w=copy.deepcopy(w_in)
    j_history=[]
    p_history=[]
    b=b_in
    w=w_in
    
    for i in range(num_iters):
        dj_dw, dj_db=gradient_function(x,y,w,b)
        
        b=b- alpha * dj_db
        w=w- alpha * dj_dw
        
        if i<100000:
            j_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
            
        if i%math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: cost {j_history[-1]:0.2e}",
                 f"dj_dw: {dj_dw: 0.3e}, dj_db:{dj_db: 0.3e}",
                 f"w:{w:0.3e},b:{b:0.5e}")
    return w,b,j_history,p_history
    


# In[14]:


w_init=0
b_init=0

iterations=1000
tmp_alpha=1.0e-2
w_final,b_final,j_hist,p_hist=gradient_descent(x_train,y_train,w_init,b_init,tmp_alpha,iterations,compute_cost,compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# In[18]:


fig,(ax1,ax2)=plt.subplots(1,2,constrained_layout=True,figsize=(12,4))
ax1.plot(j_hist[:100])
ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[19]:


print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")


# In[20]:


fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)


# In[21]:


fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)


# In[ ]:

