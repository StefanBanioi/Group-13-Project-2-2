import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

dataset = pd.read_csv('Data\healthinsurance2.csv')
dataset = dataset.dropna()

dataset = dataset.drop(columns=['city', 'job_title'])

X = dataset.drop(columns=['claim'])
y = dataset['claim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

# specific parameters
params = {
    "learning_rate": 0.15,
    "loss": "huber",
    "min_samples_leaf": 1,
    "min_samples_split": 8,
    "max_depth": 7,
    "subsample": 0.85,
    "n_estimators": 695
}

model = GradientBoostingRegressor(**params, random_state=69)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Root Mean Squared Error (Test):", rmse)
print("Mean Absolute Error (Test):", mae)
print("Mean Absolute Percentage Error (Test):", mape)

y_pred_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

print("Root Mean Squared Error (Train):", rmse_train)
print("Mean Absolute Error (Train):", mae_train)
print("Mean Absolute Percentage Error (Train):", mape_train)
