#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[13]:


import pandas as pd

# Load the CSV file into a DataFrame
mydata = pd.read_csv('Medicalpremium.csv')
mydata.describe()


# In[14]:


# The input features and output features
X = mydata.drop(['PremiumPrice'], axis = 1)
y = mydata['PremiumPrice']
X.shape


# In[15]:


print(X[:10])


# In[18]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np 
import matplotlib.pyplot as plt


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.fit_transform(X_test)

param_dist = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

# Instantiation of the model
rand_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=20),
    param_distributions=param_dist,
    n_iter=100,  # Set the number of iterations to 1
    cv=5
)
rand_search.fit(X_train_scaled, y_train)

best_params = rand_search.best_params_
print("Best Parameters:", best_params)

# Predictions using the best model
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)



# In[19]:


# The training r_sq
print('The training rsq is: %.2f'% rand_search.score(X_train_scaled, y_train))

# Prediction on the training dataset
ytrain_pred = rand_search.predict(X_train_scaled)
print('The MAE is: %.2f'% mean_absolute_error(y_train, ytrain_pred))
print('The MSE is: %.2f'% mean_squared_error(y_train, ytrain_pred))

print('The RMSE is: %.2f'% np.sqrt(mean_squared_error(y_train, ytrain_pred)))

# Prediction on the testing data
ytest_pred = rand_search.predict(X_test_scaled)

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


# In[20]:


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




