import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 

# Path to the CSV file
csv_file_path = 'healthinsurance2.csv'  # Replace with the actual path to your CSV file
# Read the CSV file
mydata = pd.read_csv(csv_file_path)
mydata = mydata.dropna()

X = mydata.drop(columns=['claim','job_title','city'])
y = mydata['claim']


# X = data.drop('PremiumPrice', axis=1)
# y = data['PremiumPrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of XGBRegressor with the given parameters
model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',  # Comment out if no GPU is available
        max_depth=9,
        gamma=0.1,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.7,
        random_state = 69)
# possibleParameters= {
#   'max_depth' : range (2, 51, 1),
#   'n_estimators'  : range (50,310,5),
#   'learning_rate' : [0.01, 0.02,0.03,0.04,0.05,0.1],
#   'subsample':  [0.65,0.7,0.75,0.8,0.85,0.95,1],
#   'gamma': [0,0.01,0.05,0.1]
# }


# rand_search = RandomizedSearchCV(
#     xgb.XGBRegressor(random_state=69),
#     param_distributions=possibleParameters,
#     n_iter=1000,  # Set the number of iterations to 1
#     cv=5, n_jobs=-1
# )
# rand_search.fit(X_train_scaled, y_train)

# best_params = rand_search.best_params_

# print("Best Parameters:", best_params)


# #best parameters


# #predictions using the best model
# model = rand_search.best_estimator_
# Fit the model to the training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(y_pred)

from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(
    base_estimator=model,
    n_estimators=10,       # Number of bootstrap samples
    bootstrap=True,        # Use bootstrapped samples
    n_jobs=-1,
    random_state = 69# Use all available cores
)

bagging_model.fit(X_train_scaled, y_train)

y_pred2 = bagging_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred2)
mse = mean_squared_error(y_test, y_pred2)
mape = mean_absolute_percentage_error(y_test, y_pred2)
print(mae)
print(math.sqrt(mse))
print(mape)

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=10, shuffle=True, random_state=69)
cv_mae = []
cv_mse = []
cv_mape = []
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the BaggingRegressor model
    bagging_model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = bagging_model.predict(X_test)

    # Compute metrics for this fold
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_mse = mean_squared_error(y_test, y_pred)
    fold_mape = mean_absolute_percentage_error(y_test, y_pred)

    cv_mae.append(fold_mae)
    cv_mse.append(fold_mse)
    cv_mape.append(fold_mape)

# Calculate mean of metrics across folds
mean_cv_mae = np.mean(cv_mae)
mean_cv_mse = np.mean(cv_mse)
mean_cv_mape = np.mean(cv_mape)


print('Mean Cross-validated MAE:', mean_cv_mae)
print('Mean Cross-validated RMSE:', np.sqrt(mean_cv_mse))
print('Mean Cross-validated MAPE:', mean_cv_mape)


import numpy as np
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=10, shuffle=True, random_state=69)
cv_mae = []
cv_mse = []
cv_mape = []
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the BaggingRegressor model
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Compute metrics for this fold
    fold_mae = mean_absolute_error(y_test, y_pred)
    fold_mse = mean_squared_error(y_test, y_pred)
    fold_mape = mean_absolute_percentage_error(y_test, y_pred)

    cv_mae.append(fold_mae)
    cv_mse.append(fold_mse)
    cv_mape.append(fold_mape)

# Calculate mean of metrics across folds
mean_cv_mae = np.mean(cv_mae)
mean_cv_mse = np.mean(cv_mse)
mean_cv_mape = np.mean(cv_mape)


print('Mean Cross-validated MAE:', mean_cv_mae)
print('Mean Cross-validated RMSE:', np.sqrt(mean_cv_mse))
print('Mean Cross-validated MAPE:', mean_cv_mape)



