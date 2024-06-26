import tkinter as tk
from adapter import Adapter
import xgboost as xgb
import pandas as pd
import numpy as np
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('xgboost_model.json')
import shap
import numpy as np
import	joblib
import pandas as pd
shap.initjs()

# Frontend class
class Frontend:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.root = tk.Tk()
        self.root.geometry("600x600")  # Increased window size
        self.root.title("Predicting Insurance Premium for Individuals (One Year)")

        # Title Label with increased font size
        self.title_label = tk.Label(self.root, text="Predicting Insurance Premium for Individuals (One Year)", font=("Helvetica", 16, "bold"))
        self.title_label.grid(row=0, columnspan=2, padx=15, pady=10)

        # Creating frames for better organization
        self.frame_labels = tk.Frame(self.root)
        self.frame_labels.grid(row=1, column=0, padx=15, pady=10)

        self.frame_entries = tk.Frame(self.root)
        self.frame_entries.grid(row=1, column=1, padx=15, pady=10)

        # Labels with increased font size
        # Old labels for the other dataset. We need to change them for the new dataset. 
        '''
        labels = [
            "Age:", "Diabetes (Yes/No):", "Blood Pressure Problems (Yes/No):",
            "Any Transplants (Yes/No):", "Chronic Diseases (Yes/No):",
            "Height (cm):", "Weight (kg):", "Known Allergies (Yes/No):",
            "History of Cancer in Family (Yes/No):", "Major Surgeries (Yes/No):"
        ]
        '''

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
                
        '''

        labels = ["Age" , "Sex (Male/Female)" , "Weight (Kg)" ,  "BMI (Value)" , "Hereditary Diseases (Choose from list)" , "No of Dependents" , "Smoker (Yes/No)" , "Blood Pressure (Yes/No)" , "Diabates (Yes/No)" , "Regular Exercise (Yes/No)" ]
        # Define the options for the 'Hereditary Diseases' field
        
        #Age
        label = tk.Label(self.frame_labels, text="Age", font=("Helvetica", 12))
        label.grid(row=0, column=0, sticky="w", pady=5)

        #Sex
        label = tk.Label(self.frame_labels, text="Sex (Male/Female)", font=("Helvetica", 12))
        label.grid(row=1, column=0, sticky="w", pady=5)

        #Weight
        label = tk.Label(self.frame_labels, text="Weight (Kg)", font=("Helvetica", 12))
        label.grid(row=2, column=0, sticky="w", pady=5)

        #BMI
        label = tk.Label(self.frame_labels, text="BMI (Value)", font=("Helvetica", 12))
        label.grid(row=3, column=0, sticky="w", pady=5)

        #Hereditary Diseases
        # Define the options for the 'Hereditary Diseases' field
        diseases = ["NoDisease", "Arthritis", "Alzheimer", "Diabetes", "Cancer", "EyeDisease", "Obesity", "High BP", "HeartDisease", "Epilepsy"]

        label = tk.Label(self.frame_labels, text="Hereditary Diseases", font=("Helvetica", 12))
        label.grid(row=4, column=0, sticky="w", pady=5)
        
        # Create a StringVar to hold the selected value
        self.disease_var = tk.StringVar(self.frame_entries)
        self.disease_var.set(diseases[0])  # set the default value

        # Create the OptionMenu
        disease_menu = tk.OptionMenu(self.frame_entries, self.disease_var, *diseases)
        disease_menu.grid(row=4, column=1, pady=5)  # adjust the row and column as needed

        #No of Dependents
        label = tk.Label(self.frame_labels, text="No of Dependents", font=("Helvetica", 12))
        label.grid(row=5, column=0, sticky="w", pady=5)

        #Smoker
        label = tk.Label(self.frame_labels, text="Smoker (Yes/No)", font=("Helvetica", 12))
        label.grid(row=6, column=0, sticky="w", pady=5)

        #Blood Pressure
        label = tk.Label(self.frame_labels, text="Blood Pressure (Value)", font=("Helvetica", 12))
        label.grid(row=7, column=0, sticky="w", pady=5)

        #Diabetes
        label = tk.Label(self.frame_labels, text="Diabates (Yes/No)", font=("Helvetica", 12))
        label.grid(row=8, column=0, sticky="w", pady=5) 

        #Regular Exercise
        label = tk.Label(self.frame_labels, text="Regular Exercise (Yes/No)", font=("Helvetica", 12))
        label.grid(row=9, column=0, sticky="w", pady=5)


        '''
        # Entries with increased size
        self.entries = []
        for i in range(len(labels)):
            entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
            entry.grid(row=i, column=0, pady=5)
            self.entries.append(entry)
        '''

        ''' 
        #Old code
        # Entries with increased size
        self.entries = []
        for i, label in enumerate(labels):
                entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
                entry.grid(row=i, column=0, pady=5)
        '''
        self.entries = []
        # Age Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=0, column=0, pady=5)
        self.entries.append(entry)

        # Sex Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=1, column=0, pady=5)
        self.entries.append(entry)

        # Weight Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=2, column=0, pady=5)
        self.entries.append(entry)

        # BMI Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=3, column=0, pady=5)
        self.entries.append(entry)

        # No of Dependents Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=5, column=0, pady=5)
        self.entries.append(entry)

        # Smoker Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=6, column=0, pady=5)
        self.entries.append(entry)

        # Blood Pressure Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=7, column=0, pady=5)
        self.entries.append(entry)

        # Diabetes Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=8, column=0, pady=5)
        self.entries.append(entry)

        # Regular Exercise Entry
        entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
        entry.grid(row=9, column=0, pady=5)
        self.entries.append(entry)
        '''
        self.entries.append(entry)
        self.button_send = tk.Button(self.root, text="Send", command=self.send_data, font=("Helvetica", 15))
        self.button_send.grid(row=2, column=0, pady=10)
        '''
        self.button_calc = tk.Button(self.root, text="Calculate", command=self.calculate_data, font=("Helvetica", 15))
        self.button_calc.grid(row=2, columnspan=2, pady=15)

        # Result Text Widget with increased font size
        self.result_text = tk.Text(self.root, height=2, width=30, font=("Helvetica", 12), state="disabled")
        self.result_text.grid(row=3, columnspan=2, pady=10)
        
    '''
    def send_data(self):
        # Get the data from the entry widget and send it to the adapter

        age = self.entries[0].get()
        diabetes = self.entries[1].get()
        blood_pressure = self.entries[2].get()
        any_transplants = self.entries[3].get()
        chronic_diseases = self.entries[4].get()
        height = self.entries[5].get()
        weight = self.entries[6].get()
        known_allergies = self.entries[7].get()
        history_of_cancer = self.entries[8].get()
        major_surgeries = self.entries[9].get()

        response = self.adapter.request(age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries)
        print(response)
        return response
    '''

    def send_data(self):
        
    # Get the data from the entry widget and send it to the adapter
        age = self.entries[0].get()
        sex = self.entries[1].get()
        weight = self.entries[2].get()
        bmi = self.entries[3].get()
        hereditary_diseases = self.disease_var.get()
        no_of_dependents = self.entries[4].get()
        smoker = self.entries[5].get()
        blood_pressure = self.entries[6].get()
        diabetes = self.entries[7].get()
        regular_exercise = self.entries[8].get()
        response = self.adapter.request(age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise)
        
        age = response[0]
        sex = response[1]
        weight = response[2]
        bmi = response[3]
        hereditary_diseases = response[4]
        no_of_dependents = response[5]
        smoker = response[6]    
        blood_pressure = response[7]
        diabetes = response[8]
        regular_exercise = response[9]
    
        # Your instance as a NumPy array
        X = np.array([age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise])
        X = X.reshape(1, -1)

        # Define your feature names
        feature_names = ['age', 'sex', 'weight', 'bmi', 'hereditary_diseases', 'no_of_dependents', 'smoker', 'bloodpressure', 'diabetes', 'regular_ex']
        
        # Convert your instance into a pandas DataFrame
        df = pd.DataFrame(X, columns=feature_names)

        # Initialize JavaScript visualization code for SHAP
        shap.initjs()

        # Calculate SHAP values
        explainer = shap.TreeExplainer(loaded_model)
        #load scaler.joblib

        scaler = joblib.load('scaler.joblib')
        df_scaled = scaler.transform(df)

        shap_values = explainer.shap_values(df_scaled)

        # Create a SHAP Explanation object
        shap_values_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=df)

        # Create a waterfall plot
        shap.plots.waterfall(shap_values_explanation[0])
        
    
        print(response)
        return response
         
    def calculate_data(self):

        result = self.adapter.calculate_data(self.send_data())
        
        # Clear the text box
        self.result_text.config(state="normal")
        self.result_text.delete('1.0', tk.END)

        # Insert the new value
        self.result_text.insert(tk.END, str(result))

         # Disable the text box so it's read-only
        self.result_text.config(state="disabled")
        print('test test')
        
        return result


    def run(self):
        self.root.mainloop()