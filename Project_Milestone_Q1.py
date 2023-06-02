#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

df = pd.read_csv("train.csv")


# In[6]:


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


df.shape


# In[3]:


import pandas as pd
testdf = pd.read_csv('test_public.csv')
testdf.shape


# In[ ]:





# In[10]:


#what is the distribution of travel time for all trips?

import matplotlib.pyplot as plt


# Compute the travel time in seconds
df['TRAVEL_TIME'] = (df['POLYLINE'].str.count('\[') - 1) * 15

# Plot the distribution of travel time
plt.hist(df['TRAVEL_TIME'], bins=1000, edgecolor='black')
plt.xlabel('Travel Time (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Travel Time')
plt.xlim(0, 6000)
plt.show()


# In[ ]:





# In[ ]:




