import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

class ANNModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('model_ANN.h5')
        self.scaler = joblib.load('scaler.pkl')
    
    def predict(self, input_data):
        # Input: input_data: 2D numpy array or pandas with all the original features
        # categorical variables should be one-hot encoded, sex is also in hot encoded form

        # Output: predictions: 1D numpy array with the predictions
        

        

        
        preprocessed_data = self.scaler.transform(input_data)
        
        # Make predictions using the loaded model
        predictions = self.model.predict(preprocessed_data)
      
        return predictions


ann_model = ANNModel(model_path)
predictions = ann_model.predict(input_data)
print(predictions)