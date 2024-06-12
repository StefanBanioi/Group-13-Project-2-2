import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Path to the CSV file
csv_file_path = 'Data\healthinsurance2.csv'  # Replace with the actual path to your CSV file
# Read the CSV file

mydata = pd.read_csv(csv_file_path)

mydata = mydata.dropna()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
mydata['city'] = le.fit_transform(mydata['city'])
X = mydata.drop(columns=['claim', 'city', 'job_title'])
y = mydata['claim']


# X = data.drop('PremiumPrice', axis=1)
# y = data['PremiumPrice']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#This was used for Hyperparameter tuning using RandomizedSearchCV(long process and commented out since we saved the parameters)

# possibleParameters= {
#   'max_depth' : range (2, 51, 1),
#   'n_estimators'  : range (50,1000,25),
#   'learning_rate' : [0.01, 0.02,0.03,0.05,0.1,0.2],
#   'subsample':  [0.65,0.7,0.75,0.8,0.85,0.95,1],
#   'colsample_bytree':[0.80,0.85,0.90,0.95,1],
# }


# rand_search = RandomizedSearchCV(
#     xgb.XGBRegressor(random_state=69),
#     param_distributions=possibleParameters,
#     n_iter=300,  # Set the number of iterations to 1
#     cv=5, n_jobs=-1
# )
# rand_search.fit(X_train_scaled, y_train)

# best_params = rand_search.best_params_

# print("Best Parameters:", best_params)


# #best parameters


# #predictions using the best model
# model = rand_search.best_estimator_
# Fit the model to the training data

# # Create an instance of XGBRegressor with the given parameters
model = xgb.XGBRegressor(
        tree_method='hist',  # Comment out if no GPU is available
        max_depth=32,
        learning_rate=0.03,
        n_estimators=475,
        subsample=0.65,
        colsample_bytree=0.85,
        random_state = 69)

model.fit(X_train_scaled, y_train)

#save model
model.save_model('xgboost_model.json')

y_pred = model.predict(X_test_scaled)
print(y_pred)
y_pred_train = model.predict(X_train_scaled)
print(y_pred_train)
'''
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape= mean_absolute_percentage_error(y_test,y_pred)

mae_train= mean_absolute_error(y_train, y_pred_train)
mse_train= mean_squared_error(y_train,y_pred_train)
mape_train= mean_absolute_percentage_error(y_train,y_pred_train)
print('train data')
print("Mae: ", mae_train)
print("RMSE: ",math.sqrt(mse_train))
print("MAPE: ",mape_train)
print('test data')

print("Mae: ", mae)
print("RMSE: ", math.sqrt(mse))
print("MAPE: ", mape)
'''


# Load the saved model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('xgboost_model.json')
y_pred = loaded_model.predict(X_test_scaled)
# Set the best_model to the loaded model
best_model = loaded_model

def predict_premium(params):
    """
    This function takes a list of parameters representing a person's medical information
    and predicts the corresponding premium price using the trained XGBoost model.

    Args:
        params: A list of integers representing medical information (e.g., age, diabetes, etc.).
            Follows the same format as the features in your training data.

    Returns:
        A float representing the predicted premium price.
    """
    # Reshape the input parameters into a 2D array for the model
    X_new = np.array([params]).reshape(1, -1)

    # Apply the same StandardScaler transformation used during training
    X_new_scaled = scaler.transform(X_new)

    # Make prediction using the trained XGBoost model
    predicted_price = best_model.predict(X_new_scaled)[0]

    return predicted_price