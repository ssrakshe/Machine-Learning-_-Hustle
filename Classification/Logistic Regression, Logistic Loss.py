#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')


# In[2]:


soup_bowl()


# In[3]:


x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)


# In[4]:


#Now, let's get a surface plot of the cost using a squared error cost:
#𝐽(𝑤,𝑏)=1/2𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2


# In[5]:


plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()


# In[6]:


plt_two_logistic_loss_curves()


# In[8]:


plt.close('all')
cst = plt_logistic_cost(x_train,y_train)


# In[9]:


print("This curve is well suited to gradient descent! It does not have plateaus, local minima, or discontinuities. Note, it is not a bowl as in the case of squared error. Both the cost and the log of the cost are plotted to illuminate the fact that the curve, when the cost is small, has a slope and continues to decline. Reminder: you can rotate the above plots using your mouse.")


# In[ ]:




