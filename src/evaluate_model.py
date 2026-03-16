#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src/evaluate_model.py
import os
import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

import warnings 
warnings.filterwarnings('ignore')

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir('D:\ShivaniG_AI_ML\capital-markets-trade-settlement-ml')


# In[3]:


os.getcwd()


# In[4]:


# Load features

df = pd.read_csv("data/trade_settlement_features.csv")


# In[5]:


df.head(5)


# In[6]:


X = df.drop(["trade_id","security_isin","settlement_failed"], axis=1)
y = df["settlement_failed"]


# In[7]:


# Load trained model
model = joblib.load("models/settlement_prediction_model.pkl")


# In[8]:


# Predict
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]


# In[9]:


# Evaluation
print("Classification Report:")
print(classification_report(y, y_pred))


# In[10]:


roc = roc_auc_score(y, y_prob)
print(f"ROC-AUC Score: {roc:.3f}")


# In[11]:


cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)


# In[ ]:




