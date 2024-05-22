#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
mydata = pd.read_csv('healthinsurance2cleaned.csv')
mydata.describe()


# In[3]:


# The input features and output features
mydata = mydata.dropna()
X = mydata.drop(columns=['claim','job_title','city'])
y = mydata['claim']
X.shape


# In[4]:


print(X[:10])


# In[5]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, mean_absolute_percentage_error
from numpy import *
import matplotlib.pyplot as plt


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.fit_transform(X_test)

model = RandomForestRegressor(n_estimators=800, max_depth=80, min_samples_split=2, min_samples_leaf=1, bootstrap = True, random_state=69)

# Fitting the model
PP_rf = model.fit(X_train_scaled, y_train)

# ultima ultima Best Parameters: {'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 80, 'bootstrap': True}
# ultima Best Parameters: {'n_estimators': 1650, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_depth': 25, 'bootstrap': True}
# Best Parameters: {'n_estimators': 1700, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 105, 'bootstrap': True}


# In[6]:


# The training r_sq
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, mean_absolute_percentage_error

print('The training rsq is: %.2f'% PP_rf.score(X_train_scaled, y_train))

# Prediction on the training dataset
ytrain_pred = PP_rf.predict(X_train_scaled)
print('The MAE is: %.2f'% mean_absolute_error(y_train, ytrain_pred))
print('The MSE is: %.2f'% mean_squared_error(y_train, ytrain_pred))

print('The RMSE is: %.2f'% np.sqrt(mean_squared_error(y_train, ytrain_pred)))

# Prediction on the testing data
ytest_pred = PP_rf.predict(X_test_scaled)

print('The MAE is: %.2f'% mean_absolute_error(y_test, ytest_pred))
print('The MSE is: %.2f'% mean_squared_error(y_test, ytest_pred))
print('The MAPE is: %.2f'% mean_absolute_percentage_error(y_test, ytest_pred))

print('The RMSE is: %.2f'% np.sqrt(mean_squared_error(y_test, ytest_pred)))

plt.figure(figsize=(10, 6))
plt.scatter(y_train, ytrain_pred, color='blue', label='Training Data')
# Scatter plot for test data
plt.scatter(y_test, ytest_pred, color='green', label='Test Data')
# Plotting diagonal line for perfect predictions
plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
         [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
         '--', color='red', label='Perfect Predictions')
plt.xlabel('Actual Premium Prices')
plt.ylabel('Predicted Premium Prices')
plt.title('Predicted vs. Actual Premium Prices')
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


# Calculate residuals
train_residuals = y_train - ytrain_pred
test_residuals = y_test - ytest_pred

# Residual plot for both training and test data
plt.figure(figsize=(10, 6))
plt.scatter(ytrain_pred, train_residuals, color='blue', label='Training Data')
plt.scatter(ytest_pred, test_residuals, color='green', label='Test Data')
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at y=0
plt.xlabel('Predicted Premium Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




