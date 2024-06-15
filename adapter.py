from backend import Backend

# Adapter class
class Adapter:
    def __init__(self, backend: Backend):
        self.backend = backend
    
    def request(self, age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise):
        return self.backend.handle_request(age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise)
    
    '''
    def request(self, **kwargs):
        # Mapping of argument names to their values age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise
        arguments = {
            "age": kwargs.get("age"),
            "sex": kwargs.get("sex"),
            "weight": kwargs.get("weight"),
            "bmi": kwargs.get("bmi"),
            "hereditary_diseases": kwargs.get("hereditary_diseases"),
            "no_of_dependents": kwargs.get("no_of_dependents"),
            "smoker": kwargs.get("smoker"),
            "blood_pressure": kwargs.get("blood_pressure"),
            "diabetes": kwargs.get("diabetes"),
            "regular_exercise": kwargs.get("regular_exercise"),
        }

        # Assuming backend.handle_request can handle partial data or has defaults
        return self.backend.handle_request(**arguments)
         def request(self, age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries):
        return self.backend.handle_request(age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries)'''
        
    def calculate_data(self, data):
        #call the calculate_premium method from the backend 
        result = self.backend.calculate_premium(data)
        return result