#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def forward_euler_1d(f, u_0, t_n, h):
    u_n = np.zeros(t_n)
    
    for i, u in enumerate(u_n):
        
        if i == 0:
            u += u_0
        else:
            u += u_n[i-1] + h*t_n[i]
    
    return (t_n, u_n)


# ### Solução Exemplo 1: Sitema Associado Reação Química

# In[53]:

u1 = 1
u2 = 0
u3 = 0
k1 = 0.04
k2 = 1e4
k3 = 3e7
init_time = 0


# In[52]:


def solution_reactions(u1, u2, u3, k1, k2, k3, init_time, final_time, h):
    
    n_steps = int((final_time-init_time)/h)
    array_t_n = np.linspace(init_time, final_time, num = n_steps)
    
    u1_n = [u1]
    u2_n = [u2]
    u3_n = [u3]

    for i in range(len(array_t_n)-1):
        
        J = np.array([[-k1, k2*u3, k2*u2],
                      [k1, -k2*u3-2*k3*u2, -k2*u2],
                      [0, 2*k3*u2, 0]])
        
        eigenvalues, _ = np.linalg.eig(J)
        min_lambda = eigenvalues.min()
        max_lambda = eigenvalues.max()
        
        if (max_lambda > 5.0e-16):
            print("Erro no auto valor:", max_lambda)
        
        h_opt = 2/abs(min_lambda)
        if h_opt < h:
            print("Erro no h:", h, ", ", h_opt)
        
        f1 = -k1*u1 + k2*u2*u3
        f2 = k1*u1 - k2*u2*u3 - k3*(u2)**2
        f3 = k3*(u2)**2

        u1 = u1 + h*f1
        u2 = u2 + h*f2
        u3 = u3 + h*f3

        u1_n.append(u1)
        u2_n.append(u2)
        u3_n.append(u3)

    array_u1_n = np.array(u1_n)
    array_u2_n = np.array(u2_n)
    array_u3_n = np.array(u3_n)

    return (array_t_n, array_u1_n, array_u2_n, array_u3_n)


# In[54]:


final_time = 1
h = 0.1
solution_error = solution_reactions(u1, u2, u3, k1, k2, k3, init_time, final_time, h)


# In[50]:


final_time = 1000
h = 0.0001
solution = solution_reactions(u1, u2, u3, k1, k2, k3, init_time, final_time, h)


# In[51]:


plt.plot(solution[0], solution[1])
plt.plot(solution[0], solution[2])
plt.plot(solution[0], solution[3])

