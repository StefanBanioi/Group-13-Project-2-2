#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


import pandas as pd

# Load the CSV file into a DataFrame
mydata = pd.read_csv('Random Forest\Medicalpremium.csv')
mydata.describe()


# In[3]:


# The input features and output features
X = mydata.drop(['PremiumPrice'], axis = 1)
y = mydata['PremiumPrice']
X.shape


# In[4]:


print(X[:10])


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np 
import matplotlib.pyplot as plt

# Splitting the dataset using a random seed value (69)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.fit_transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Instantiation of the model
model = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=20)

# Fitting the model
PP_rf = model.fit(X_train_scaled, y_train)

# The training r_sq
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


# In[6]:


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

