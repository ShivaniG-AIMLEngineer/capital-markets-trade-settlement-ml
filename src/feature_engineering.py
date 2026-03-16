#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Feature Engineering for Trade Settlement Failure Prediction
# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir('D:\ShivaniG_AI_ML\capital-markets-trade-settlement-ml')


# In[3]:


os.getcwd()


# In[4]:


# Load cleaned data
df = pd.read_csv(r"data\trade_settlement_clean.csv")


# In[5]:


# Feature: Trade size category
bins = [0, 50000, 500000, 2000000, 5000000]
labels = ["Small","Medium","Large","XL"]
df["trade_size_category"] = pd.cut(df["trade_amount"], bins=bins, labels=labels)

# One-hot encoding
df = pd.get_dummies(df, columns=["security_type","currency","counterparty","trade_size_category"], drop_first=True)

# Save features
df.to_csv(r"data\trade_settlement_features.csv", index=False)
print(r"Feature engineering completed. Features saved to ..data\trade_settlement_features.csv")


# In[ ]:




