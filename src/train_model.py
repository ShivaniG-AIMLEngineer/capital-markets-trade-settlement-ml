#!/usr/bin/env python
# coding: utf-8

# In[2]:


# src/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# In[3]:


os.chdir('D:\ShivaniG_AI_ML\capital-markets-trade-settlement-ml')


# In[4]:


os.getcwd()


# In[5]:


# Load features
df = pd.read_csv(r"data\trade_settlement_features.csv")


# In[6]:


df.shape


# In[7]:


# Define features & target
X = df.drop(["trade_id","security_isin","settlement_failed"], axis=1)
y = df["settlement_failed"]


# In[10]:


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)


# In[12]:


# Save trained model
joblib.dump(model, "models/settlement_prediction_model.pkl")


# In[14]:


print("Model trained and saved to models/settlement_prediction_model.pkl")


# In[ ]:





# In[ ]:




