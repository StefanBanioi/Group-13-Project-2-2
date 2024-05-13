import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow.keras.layers as layers

# Load the data
def load_data(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, file_name)
    data = pd.read_csv(data_file)
    return data

# Preprocess the data
def preprocess_data(data):
    X = data.drop('PremiumPrice', axis=1)
    y = data['PremiumPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Build the model
def build_model(input_nodes, hidden_nodes, output_nodes):
    model = tf.keras.models.Sequential([
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu, input_shape=(input_nodes,)),
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu),
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu),
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu),
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu),
        layers.Dense(hidden_nodes, activation=tf.nn.leaky_relu),
        layers.Dense(output_nodes)
    ])
    model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='AdamW'), metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    return model

# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_mae, train_rmse = model.evaluate(X_train, y_train, verbose=0)[1:]
    print("Mean Absolute Error on training data: ", train_mae)
    print("Root Mean Squared Error on training data: ", train_rmse)
    test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)[1:]
    print("Mean Absolute Error on test data: ", test_mae)
    print("Root Mean Squared Error on test data: ",test_rmse)

# Predict with the model
def predict_model(model, X_train, X_test, y_train, y_test):
    predictions2 = model.predict(X_train[:5])
    print("Train-set evaluation: ")
    print("Predicted values are: ", predictions2)
    print("Real values are: ", y_train[:5])
    predictions = model.predict(X_test[:5])
    print("Test-set evaluation: ")
    print("Predicted values are: ", predictions)
    print("Real values are: ", y_test[:5])
    predictions2 = model.predict(X_train[:5])

# Main function to run the code
def main():
    data = load_data('Medicalpremium.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Perform parameter tuning using GridSearchCV
    param_grid = {
        'input_nodes': 10,
        'hidden_nodes': [10, 20, 30],
        'output_nodes': [1]
    }
    model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    
    # Build the model with the best parameters
    model = build_model(input_nodes=best_params['input_nodes'], hidden_nodes=best_params['hidden_nodes'], output_nodes=best_params['output_nodes'])
    
    # Train the model
    train_model(model, X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Predict with the model
    predict_model(model, X_train, X_test, y_train, y_test)
    
    model.save('model_parameters.h5')

main()
