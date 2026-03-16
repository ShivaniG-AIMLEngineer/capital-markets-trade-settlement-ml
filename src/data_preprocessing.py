#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Data Preprocessing for Trade Settlement Failure Prediction
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

#get_ipython().run_line_magic('matplotlib', 'inline')



# In[27]:


os.chdir(r'D:\ShivaniG_AI_ML\capital-markets-trade-settlement-ml')


# In[28]:


os.getcwd()


# In[36]:


# Load feature-engineered dataset
df = pd.read_csv(r'data\trade_settlement_dataset.csv')


# In[37]:


df.head(5)


# In[38]:


df.duplicated().sum()


# In[39]:


df.info()


# In[40]:


df.isnull().sum()


# In[41]:


df.shape


# In[26]:


# Filter invalid trades 
df = df[df["trade_amount"] > 0]


# In[43]:


# Save cleaned data
df.to_csv(r"data\trade_settlement_clean.csv", index=False)


# In[46]:


print(r"Data preprocessing completed. Clean file saved to ..data\trade_settlement_clean.csv")


# In[ ]:




