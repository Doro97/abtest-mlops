#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('responded_users.csv')
#df=df.drop(columns='Unnamed: 0')
df


# In[3]:


#users who responded yes rep by 1 and those who responded no by 0
df.loc[(df['yes']==1),'responses']=1
df['responses']=df['responses'].fillna(0)
df


# In[4]:


#drop platform_os column
new_df=df.drop(columns='platform_os')
new_df


# In[5]:


#new_df=new_df.drop(columns='Unnamed: 0')
new_df=new_df.drop(columns='date')
new_df


# In[ ]:





# In[6]:


new_df.info()


# We are going to use three different models for analysis. We will find the score for every split and then take average to get the overall score

# In[7]:


new_df.columns


# In[8]:


#select columns with a device_make value count of more than 10
#print(new_df['device_make'].value_counts(ascending=False))
sub_df = new_df[new_df.groupby('device_make').device_make.transform('count')>10].copy() 
print(sub_df['device_make'].value_counts(ascending=False))


# In[9]:


sub_df=sub_df.set_index('auction_id')
sub_df


# In[10]:


cat_vars=['experiment', 'hour', 'device_make', 'browser', 'day_of_the_week']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(sub_df[var], prefix=var)
    data1=sub_df.join(cat_list)
    sub_df=data1
cat_vars=[ 'experiment', 'hour', 'device_make', 'browser', 'day_of_the_week']
data_vars=sub_df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[11]:


data_final=sub_df[to_keep]
data_final.columns.values
cols=data_final.columns


# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression 
# from sklearn import metrics
# X = data_final.loc[:, data_final.columns != 'responses']
# y = data_final.loc[:, data_final.columns == 'responses']
# from imblearn.over_sampling import SMOTE
# os = SMOTE(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# columns = X_train.columns
# os_data_X,os_data_y=os.fit_sample(X_train, y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
# os_data_y= pd.DataFrame(data=os_data_y,columns=['responses'])
# #we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("Number of no response in oversampled data",len(os_data_y[os_data_y['responses']==0]))
# print("Number of responses",len(os_data_y[os_data_y['responses']==1]))
# print("Proportion of no response data in oversampled data is ",len(os_data_y[os_data_y['responses']==0])/len(os_data_X))
# print("Proportion of response data in oversampled data is ",len(os_data_y[os_data_y['responses']==1])/len(os_data_X))
# os_data_X

# data_final_vars=data_final.columns.values.tolist()
# y=['responses']
# X=[i for i in data_final_vars if i not in y]
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# rfe = RFE(logreg, 20)
# rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
# print(rfe.support_)
# rfe.ranking_

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X=data_final[cols]
y=data_final['responses']

#Split the data into 70% training, 20% validation, and 10% test sets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[14]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[15]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[24]:


# get feature importance
importance = logreg.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
import matplotlib.pyplot as plt
# plot feature importance
#plt.bar([x for x in range(len(importance))], importance)
#plt.show()


# In[ ]:





# # Decision Tree

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

tree=DecisionTreeClassifier(max_depth=10,random_state=42)
tree.fit(X_train,y_train)

#measures accuracy
predictions=tree.predict(X_train)
score=accuracy_score(y_train,predictions)
score


# In[21]:


#Lets split the data into 5 folds.  
# We will use this 'kf'(KFold splitting stratergy) object as input to cross_val_score() method
kf =KFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X, y):
   print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
   cnt += 1


# In[ ]:


def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')


# In[ ]:


score = cross_val_score(tree.DecisionTreeRegressor(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error")
print(f'Scores for each fold: {score}')
rmse(score.mean())


# In[ ]:



#importance of the features
feature_importance=pd.DataFrame(tree.feature_importances_ ,index=cols,columns=['Importance']).sort_values('Importance',ascending=False)
feature_importance.head()


# In[26]:


# get importance
importance = tree.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[ ]:


from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor


# In[ ]:


pip install xgboost


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




