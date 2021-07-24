#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np


# In[101]:


df=pd.read_csv('C:\Users\user.DESKTOP-OMQ89VA\Desktop\Projects\abtest-mlops\data\AdSmartABdata.csv', sep=',')
df.head()


# # Exploratory Data Analysis

# In[102]:


df.info()


# In[103]:


#adding a response column (1-user responded, 0-user did not respond)
df.loc[(df['yes']==1)|(df['no']==1),'response']=1
df['response']=df['response'].fillna(0)
df


# In[104]:


# number of data points
print(f" Number of rows:  {df.shape[0]} \n Number of columns: {df.shape[1]} ")


# In[105]:


#number of unique values per column
unique_values=pd.DataFrame(df.apply(lambda x: len(x.value_counts(dropna=False)), axis=0), 
                           columns=['Unique Value Count']).sort_values(by='Unique Value Count', ascending=True)
unique_values


# In[106]:


# the percentage of missing values in the dataset
def missing_values(x):

    # Total number of elements in the dataset
    totalCells = x.size

    #Number of missing values per column
    missingCount = x.isnull().sum()

    #Total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")

missing_values(df)


# # Analysis of categorical variables 

# In[107]:


#distibution of users in each group
import matplotlib.pyplot as plt
import seaborn as sns

df['experiment'].value_counts()


# In[108]:



plt.figure(figsize=(8,5))
sns.countplot(x='experiment',data=df,palette='rainbow')
plt.title('Count of users in each experiment')
plt.show()


# The number of users have been equally distributed between the exposed group and the experiment group

# In[142]:


plt.figure(figsize=(8,5))
sns.countplot(x='response',data=df)

plt.title('Count of users who responded')
plt.show()


# In[ ]:





# In[149]:


plt.figure(figsize=(8,5))
sns.countplot(x='experiment',hue='yes',data=df,palette='rainbow')

plt.title('Count of users in each experiment separated by a yes response')
plt.show()


# In[110]:


plt.figure(figsize=(8,5))
sns.countplot(x='experiment',hue='no',data=df,palette='rainbow')

plt.title('Count of users in each experiment separated by a no response')
plt.show()


#  From the above plots , it can be seen that a very small propotion of users actually responded to the question whether or not they are aware of the brand.

# In[150]:


#distribution of browsers
plt.figure(figsize=(8,5))
sns.countplot(x='browser',data=df,palette='rainbow')
plt.title('Count of browsers')
plt.xticks(rotation=90)
plt.show()


# In[112]:


#distribution of operating system
plt.figure(figsize=(8,5))
sns.countplot(x='platform_os',data=df,palette='rainbow')
plt.title('Count of operating systems')
plt.xticks(rotation=45)
plt.show()


# In[113]:


#distribution of hour of day

plt.figure(figsize=(8,5))
sns.countplot(x='hour',data=df)
plt.title('Distribution of hour of day')
plt.xticks(rotation=45)
plt.show()


#  Most of the users responded at hour 15 and the least number of users at hour 23

# In[114]:


df['date']=pd.to_datetime(df.date)
df['day_of_the_week']=df['date'].dt.day_name()
#distribution of day of the week
plt.figure(figsize=(8,5))
sns.countplot(x='day_of_the_week',data=df)
plt.title('Distribution of days')
plt.xticks(rotation=45)
plt.show()


#  The most responses have been recorded on a Friday while the least is on a Tuesday

# In[115]:


plt.figure(figsize=(20,20))
sns.countplot(x='day_of_the_week',hue='hour',data=df,palette='rainbow')

plt.title('Responses per day in each hour')
plt.show()


#  Responses on Friday at hour 15 are the most while on Monday there are few to no responses past hour 10

# # Hypothesis testing

# The null hypothesis states that there is no difference in brand awareness between the exposed and control groups in the current case.  The level of significance is set at 0.05

# ### Using the z-test to calculate the p-value

# In[116]:


import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import ceil
get_ipython().run_line_magic('matplotlib', 'inline')

from statsmodels.stats.proportion import proportions_ztest,proportion_confint


# ### Choosing the sample size
# 
# The sample size is estimated using Power Analysis. This depends on the power of the test,the alpha value and the effect size

# In[136]:


effect_size=sms.proportion_effectsize(0.20,0.25)
required_n=sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.05,
    ratio=1)
required_n=ceil(required_n)
required_n


# In[137]:


#random sampling from the dataset to abtain a sample size of 1092
control_sample=df[df['experiment']=='control'].sample(n=required_n, random_state=22)
exposed_sample=df[df['experiment']=='exposed'].sample(n=required_n, random_state=22)

ab_test=pd.concat([control_sample,exposed_sample],axis=0)
ab_test.reset_index(drop=True, inplace=True)


# In[138]:


ab_test


# #calculating the sample size
# def sample(N,cl,e,p):
#     #calculate the z-score
#     z=stats.norm.ppf(1-(1-cl)/2)
#     #calculate n_0 value
#     n_0=z**2*p*(1-p)/e**2
#     #calculate n
#     n=n_0/(1+(n_0-1)/N)
#     #rounding up
#     n=ceil(n)
#     return n
# sample_size=sample(8076,0.95,0.05,0.5)
# sample_size

# In[139]:


conversion_rates=ab_test.groupby('experiment')['response']
#standard deviation of the proportion
std_p=lambda x: np.std(x,ddof=0)
#standard error of the proportion
se_p=lambda x:stats.sem(x,ddof=0)

conversion_rates=conversion_rates.agg([np.mean,std_p,se_p])
conversion_rates.columns=['conversion_rate','std_deviation','std_error']
conversion_rates.style.format('{:.3f}')


# In[119]:


control_results=ab_test[ab_test['experiment']=='control']['response']
exposed_results=ab_test[ab_test['experiment']=='exposed']['response']

n_con=control_results.count()
n_exp=exposed_results.count()
successes=[control_results.sum(),exposed_results.sum()]
nobs=[n_con, n_exp]


# In[120]:


z_stat,pval=proportions_ztest(successes,nobs=nobs)
z_stat,pval


# Since the obtained  p-value is 0.043 , which is lower than the alpha of 0.05, the null hypothesis is rejected .
# This means that there is a significant difference between the control group and the exposed group

# In[ ]:





# # ML modelling with MLOps
# 

# Split data by browser and platform_os, and version each split as a new version of the data in dvc.
# 

# In[159]:


df


# In[162]:


#selecting only the users who responded
new_df=df.loc[df['response']!=0]
new_df.shape


# In[163]:


unique_values=pd.DataFrame(new_df.apply(lambda x: len(x.value_counts(dropna=False)), axis=0), 
                           columns=['Unique Value Count']).sort_values(by='Unique Value Count', ascending=True)
unique_values


# In[177]:


#converting the new dataframe to a csv file
new_df.to_csv('responded_users.csv')


# ## Splitting by platform_os

# In[172]:



new_df['platform_os'].value_counts()


# In[176]:


#selecting only the users who use platform_os labeled 6
platform_os6=new_df.loc[new_df['platform_os']==6]
platform_os6.to_csv('platform_os6.csv')
platform_os6.head()


# In[174]:


#selecting only the users who use platform_os labeled 5
platform_os5=new_df.loc[new_df['platform_os']==5]
platform_os5.to_csv('platform_os5.csv')
platform_os5


# ## Splitting by browsers

# In[178]:


new_df['browser'].value_counts()


# In[180]:


ChromeMobile=new_df.loc[new_df['browser']=='Chrome Mobile']
ChromeMobile.to_csv('Chrome Mobile.csv')
ChromeMobile


# In[181]:


ChromeMobileWebView=new_df.loc[new_df['browser']=='Chrome Mobile WebView']
ChromeMobileWebView.to_csv('Chrome_Mobile_WebView.csv')
ChromeMobileWebView


# In[182]:


Facebook=new_df.loc[new_df['browser']=='Facebook']
Facebook.to_csv('Facebook.csv')
Facebook


# In[183]:


Samsung_Internet=new_df.loc[new_df['browser']=='Samsung Internet']
Samsung_Internet.to_csv('Samsung_Internet.csv')
Samsung_Internet


# In[ ]:




