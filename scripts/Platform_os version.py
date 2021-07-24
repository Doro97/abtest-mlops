#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('responded_users.csv')
df=df.drop(columns='Unnamed: 0')
df=df.set_index('auction_id')
df


# In[4]:


#users who responded yes rep by 1 and those who responded no by 0
df.loc[(df['yes']==1),'responses']=1
df['responses']=df['responses'].fillna(0)
df


# In[5]:


#drop response column
df=df.drop(columns='response')
df


# In[7]:


#drop browser column
new_df=df.drop(columns='browser')
new_df


# In[8]:


#drop date column
new_df=new_df.drop(columns='date')
new_df


# In[9]:


new_df.info()


# In[ ]:





# In[10]:


new_df.columns


# In[11]:


#numerical variables
cols_num= ['hour','platform_os', 'yes', 'no']
#categorical variables
cols_cat=['experiment','device_make','day_of_the_week']


# In[12]:


df[cols_cat]


# In[13]:


#one-hot encoding
cols_new_cat=pd.get_dummies(df[cols_cat],drop_first=True)
cols_new_cat


# In[14]:


#add the one-hot encoded columns to the dataframe
new_df=pd.concat([new_df,cols_new_cat],axis=1)
new_df


# In[15]:


all_cols_cat=list(cols_new_cat.columns)
all_cols_cat


# In[36]:


X=new_df[all_cols_cat]
X.shape


# In[18]:


cols_new_num=['hour','platform_os', 'yes', 'no']
new_df[cols_new_num]


# In[19]:


cols_input=cols_new_num+all_cols_cat

cols_input


# In[20]:


len(cols_input)


# In[21]:


df_data=new_df[cols_input+['responses']]
df_data.head()


# In[22]:


#shuffle the samples

df_data=df_data.sample(n=len(df_data))
df_data=df_data.reset_index(drop=True)


# In[23]:


#30% of validation and test samples
df_valid_test=df_data.sample(frac=0.3)
#splits test and validation samples by 2/3
df_test=df_valid_test.sample(frac=2/3) 
df_validation=df_valid_test.drop(df_test.index)


# In[24]:


#training sample
df_train=df_data.drop(df_valid_test.index)


# # Decision Tree
# 

# In[53]:


# Split dataset into training set and test set
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

X=new_df[all_cols_cat]
y=df_data['responses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[54]:


#importance of the features
feature_importance=pd.DataFrame(tree.feature_importances_ ,index=cols_input,columns=['Importance']).sort_values('Importance',ascending=False)
feature_importance


# # Logistic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[56]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[57]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# This means the model has (139+56) correct predictions and (11+67)incorrect predictions

# In[52]:


print(classification_report(y_test, y_pred))


# In[ ]:


#feature importance


# # XGBoost

# In[58]:


import xgboost as xgb


# In[ ]:


pip install xgboost 


# In[ ]:




