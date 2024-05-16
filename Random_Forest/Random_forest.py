#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[13]:


import pandas as pd

# Load the CSV file into a DataFrame
mydata = pd.read_csv('Random_forest\Medicalpremium.csv')
#mydata = pd.read_csv('Random_Forest\healthinsurance2.csv')
mydata.describe()

#X = mydata.drop(['claim'], axis = 1)
#X = mydata.drop( ['claim'] , axis = 1)
#y = mydata['claim']
#X.shape

# Load the CSV file into a DataFrame
df = pd.read_csv('Random_Forest\healthinsurance2.csv')

# Get the unique city and count them
unique_city = df['city'].unique()
count = len(unique_city)

print(f"There are {count} unique cities in the dataset")


# In[14]:


# The input features and output features
 
X = mydata.drop(['PremiumPrice'], axis = 1)
y = mydata['PremiumPrice']
X.shape


# DO NOT delete this lines of comments
'''
I have changed the column of sex and changed all the male to 1 and the female to 0. I did this by changing all acourances 
I have changed the collom hereditary_diseases as follow: 
* NoDisease = 0
* Arthritis = 1
* Alzheimer = 2
* Diabetes = 3
* Cancer = 4
* EyeDisease = 5
* Obesity = 6
* High BP = 7
* HeartDisease = 8
* Epilepsy = 9


I have changed the collom job_title as follow:
* Student = 0
* Actor = 1
* Academician = 2
* Engineer = 3
* Chef = 4
* HomeMakers = 5
* Dancer = 6
* DataScientist = 7
* Singer = 8
* Doctor = 9
* Manager = 10
* Photographer = 11
* CA = 12
* Beautician = 13
* Accountant = 14
* CEO = 15
* HouseKeeper = 16
* Police = 17
* Lawyer = 18
* FilmMaker = 19
* Politician = 20
* ITProfessional = 21
* Architect = 22
* Technician = 23
* FashionDesigner = 24
* Buisnessman = 25
* GovEmployee = 26
* FilmDirector = 27
* Farmer = 28
* Journalist = 29
* Analyst = 30
* Blogger = 31
* Labourer = 32
* DefencePersonnels = 33
* Clerks = 34
'''






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
    'max_depth': list(range(10, 110, 10)) + [None],   #range(start, stop, step) + [None] adds None to the list of values to be tested for max_depth
    'min_samples_leaf': list(range(1, 5)),
    'min_samples_split': list(range(2, 11, 3)),
    'n_estimators': list(range(200, 2100, 200))}

# Instantiation of the model
rand_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=20),
    param_distributions=param_dist,
    n_iter=1,  # Set the number of iterations to 1
    cv=5
)
rand_search.fit(X_train_scaled, y_train)

best_params = rand_search.best_params_
print("Best Parameters:", best_params)

# Predictions using the best model
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)





# In[]

def predict_premium(params):
  """
  This function takes a list of parameters representing a person's medical information
  and predicts the corresponding premium price using the trained Random Forest model.

  Args:
      params: A list of integers representing medical information (e.g., age, diabetes, etc.).
          Follows the same format as the features in your training data.

  Returns:
      A float representing the predicted premium price.
  """

  # Reshape the input parameters into a 2D array for the model
  X_new = np.array([params]).reshape(1, -1)

  # Apply the same MinMaxScaler transformation used during training
  X_new_scaled = scaler.transform(X_new)

  # Make prediction using the trained Random Forest model
  predicted_price = best_model.predict(X_new_scaled)[0]

  return predicted_price





