from sklearn.preprocessing import StandardScaler

def predict(user_input, model):
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(np.array(user_input).reshape(-1, 1))
    prediction = model.predict(user_input_scaled)
    return prediction



import argparse
import numpy as np
import json
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV
# random search for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
import time
from datetime import datetime 
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
logging.basicConfig(filename='output2.log', level=logging.INFO, format='%(message)s')
logging.basicConfig(filename='output2.log', level=logging.error, format='%(message)s')
logging.basicConfig(filename='full.log', level=logging.warning, format='%(message)s')
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f"Current date and time: {current_date}")
start_time = time.time()
logger.info(f"Start time: {start_time}")


# Create a logger
logger = logging.getLogger('my_logger')

# Set the level of this logger. This level acts as a threshold. 
# Any message below this level will be ignored
logger.setLevel(logging.DEBUG)

# Create a file handler
handler = logging.FileHandler('output.log')

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# Now you can log to the file using
logger.info('This is a log message.')
logging.getLogger().disabled = False

def log_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function {func.__name__} with args {args} and kwargs {kwargs}")
        return func(*args, **kwargs)
    return wrapper
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a machine learning model with optional RFE and Grid Search.')
    parser.add_argument('--use_rfe', type=int, default=False, help='Use Recursive Feature Elimination (RFE)')
    parser.add_argument('--use_grid_search', type=int, default=False, help='Use Grid Search')
    param_grid = {
        'm__C': [1, 10, 100],
        'm__epsilon': [ 1,10,100],
        'm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'm__degree': [2, 3, 4],
        'm__gamma': ['scale', 'auto']
    }
    parser.add_argument('--param_grid', type=dict, default=param_grid, help='Parameter grid for Grid Search')
    return parser.parse_args()
@log_args
def main():
    args = parse_arguments()
    param_grid = args.param_grid
    df = pd.read_csv('dataset.csv')
    # drop mising values
    df = df.dropna()
    df = df.drop('city',axis=1)
    X=df.drop('claim',axis=1)
    y=df['claim']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    svr = SVR()
    estimator=SVR(kernel="linear")
    rfecv = RFECV(estimator=SVR(kernel="linear"))
    pipeline = Pipeline(steps=[('s',rfecv),('m',svr)])
    selector = RFE(estimator, n_features_to_select=5, step=1)
    grid_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, random_state=69,
                                     n_iter=100, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3, refit=False,)
    if args.use_rfe:
        logger.info("Using RFE")
        if args.use_grid_search:
            logger.info("Using Grid Search")
            grid_search.fit(X_train, y_train)
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Train Score: {grid_search.score(X_train, y_train)}")
            logger.info(f"Test Score: {grid_search.score(X_test, y_test)}")
            y_pred = grid_search.predict(X_test)
            summary = calculate_summary(y_test, y_pred)
            logger.info(f"matrix summary: {summary}")
            # Save the model
            joblib.dump(grid_search, 'grid_search_and_RFE_model.pkl')
            Tclose(start_time)
        else:
            rfecv.fit(X_train, y_train)
            logger.info(f"Number of features selected: {rfecv.n_features_}")
            logger.info(f"Train Score: {rfecv.score(X_train, y_train)}")
            logger.info(f"Test Score: {rfecv.score(X_test, y_test)}")
            y_pred = rfecv.predict(X_test)
            # Get the mask of selected features
            feature_names = df.columns
            support_mask = rfecv.support_
            logger.info(f"Support Mask: {support_mask}")
            # Use the mask to get the selected feature names
            selected_features = [feature for feature, is_selected in zip(feature_names, support_mask) if is_selected]

            print("Selected features:", selected_features)
            summary = calculate_summary(y_test, y_pred)
            logger.info(f"matrix summary: {summary}")
            # Save the model
            joblib.dump(rfecv, 'using_RFE_model.pkl')
            Tclose(start_time)
    else:
        if args.use_grid_search:
            logger.info("Using Grid Search")
            grid_search = RandomizedSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Train Score: {grid_search.score(X_train, y_train)}")
            logger.info(f"Test Score: {grid_search.score(X_test, y_test)}")
            y_pred = grid_search.predict(X_test)
            summary = calculate_summary(y_test, y_pred)
            logger.info(f"matrix summary: {summary}")
            # Save the model
            joblib.dump(grid_search, 'Not_using_RFE_but_using_Grid_Search_model.pkl') 
            Tclose(start_time)      
        else:
            logger.info("Not using RFE or Grid Search")
            logger.info(f"SVR Model: {svr.get_params()}") 
            svr.fit(X_train, y_train,)
            logger.info(f"Train Score: {svr.score(X_train, y_train)}")
            logger.info(f"Test Score: {svr.score(X_test, y_test)}")
            y_pred = svr.predict(X_test)
            summary = calculate_summary(y_test, y_pred)
            logger.info(f"matrix summary: {summary}")
            # Save the model
            joblib.dump(grid_search, 'Not_using_RFE_or_Grid_Search_model.pkl')
            Tclose(start_time)
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
def Tclose(start_time):
    end_time = time.time()
    logger.info(f"End time: {end_time}")
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")
    logger.info("Finished")
    logger.info("---------------------------------------------------")
#lod the model from the file  joblib.dump
def load_model(file_name):
    model = joblib.load(file_name)
    return model

def predict(user_input, model):
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(np.array(user_input).reshape(-1, 1))
    prediction = model.predict(user_input_scaled)
    return prediction
if __name__ == "__main__":
    main()