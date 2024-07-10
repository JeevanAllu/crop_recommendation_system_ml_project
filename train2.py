#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('crop_recommendation.csv')
print(dataset.head())

X = dataset[['N','P','K','temperature','humidity','ph','rainfall']].values
y = dataset['label'].values
print('-'*80)
print(f'Shape of X is {X.shape}\nShape of y is {y.shape}')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('-'*80)
print(f"Lenght of X_train: {len(X_train)}\nLenght of X_test: {len(X_test)}")
print(f"Lenght of y_train: {len(y_train)}\nLenght of y_test: {len(y_test)}")


# In[2]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
for i in range(10):
  print(X_train[i])
print('-'*80)
for i in range(10):
  print(X_test[i])


# ## **Alogrithms**

# 1. Support Vector Classifer (SVC)
# 

# In[3]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC()
print(classifier)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy on the Test set results                              
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_pred)))


# 2. Logistics Regression

# In[4]:


# Fitting SVM to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
print(classifier)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy on the Test set results                              
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_pred)))


# 3. Naive Bayes

# In[5]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
print(classifier)
classifier.fit(X_train, y_train)

5# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy on the Test set results                              
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_pred)))


# 4. Decision Tree Classifier

# In[6]:


# Fitting Naive Bayes to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
print(classifier)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy on the Test set results                              
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_pred)))


# 5. Random Forest Classifier

# In[7]:


# Fitting Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
print(classifier)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy on the Test set results                              
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_pred)))

