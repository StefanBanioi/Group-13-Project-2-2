import argparse
import numpy as np
import json
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
from datetime import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import logging
import time
from datetime import datetime 
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
logging.basicConfig(filename='output1.log', level=logging.INFO, format='%(message)s')
logging.basicConfig(filename='output1.log', level=logging.error, format='%(message)s')
logging.basicConfig(filename='full1.log', level=logging.warning, format='%(message)s')
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f"Current date and time: {current_date}")
start_time = time.time()
logger.info(f"Start time: {start_time}")


logging.basicConfig(filename='full.log', level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)
current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f"Current date and time: {current_date}")
start_time = time.time()
logger.info(f"Start time: {start_time}")
def calculate_summary(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


def main():
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    # drop missing values
    df = df.dropna()
    df = df.drop('city',axis=1)
    X=df.drop('claim',axis=1)
    y=df['claim']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Define and fit the model
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)

    # Predict the test set results
    y_pred = linear_regression.predict(X_test)


    # Calculate and print the summary
    summary = calculate_summary(y_test, y_pred)
    logger.info(f"Summary: {summary}")
    joblib.dump(linear_regression, 'linear_regression_model.pkl')
# Call the main function
if __name__ == "__main__":
    main()

end_time = time.time()
runtime = end_time - start_time
logger.info(f"Script has been running for {runtime} seconds")


# Save the model

