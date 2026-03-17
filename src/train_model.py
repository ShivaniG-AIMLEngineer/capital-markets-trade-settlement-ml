#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml


# In[2]:


os.chdir('D:\ShivaniG_AI_ML\capital-markets-trade-settlement-ml')


# In[3]:


os.getcwd()


# In[4]:


# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# In[6]:


# Load data
df = pd.read_csv(config["data"]["feature_data_path"])


# In[7]:


df.shape


# In[8]:


# Define features & target
X = df.drop(["trade_id","security_isin","settlement_failed"], axis=1)
y = df["settlement_failed"]


# In[9]:


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)


# In[12]:


# Save trained model
joblib.dump(model, config["output"]["model_path"])


# In[13]:


print("Model trained and saved to models/settlement_prediction_model.pkl")

