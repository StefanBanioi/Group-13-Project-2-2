# Backend class
import random


class Backend:
    def handle_request(self, age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries):
        # Process the data and return a response (value of how much the user has to pay for the insurance)

        # Only age height and weight are numerical values, the rest are strings of yes and no
        # Check if the data is valid and return an error message if it's not 
        # These are the following entry fields: age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries
        if age.isdigit() == False:
            return 'Age must be a number'
        if diabetes.lower() not in ['yes', 'no']:
            return 'Diabetes is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if blood_pressure.lower() not in ['yes', 'no']:
            return 'Blood pressure problem is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if any_transplants.lower() not in ['yes', 'no']:
            return 'Any transplants is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if chronic_diseases.lower() not in ['yes', 'no']:
            return 'Chronic diseases is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if height.isdigit() == False:   
            return 'Height must be a number in cm (no letters or special characters)'   
        if weight.isdigit() == False:   
            return 'Weight must be a number in kg (no letters or special characters)'   
        if known_allergies.lower() not in ['yes', 'no']:
            return 'Known allergies is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if history_of_cancer.lower() not in ['yes', 'no']:
            return 'History of cancer in family is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if major_surgeries.lower() not in ['yes', 'no']:
            return 'Major surgeries is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        # If the data is valid, return the response
        
        # Before returning the response, convert the strings to numerical values
        # The strings are converted to 1 if the answer is yes and 0 if the answer is no
        diabetes = 1 if diabetes.lower() == 'yes' else 0
        blood_pressure = 1 if blood_pressure.lower() == 'yes' else 0
        any_transplants = 1 if any_transplants.lower() == 'yes' else 0
        chronic_diseases = 1 if chronic_diseases.lower() == 'yes' else 0
        known_allergies = 1 if known_allergies.lower() == 'yes' else 0
        history_of_cancer = 1 if history_of_cancer.lower() == 'yes' else 0
        major_surgeries = 1 if major_surgeries.lower() == 'yes' else 0

        # We also want to put everything in a list so we can use it in the model 
        data_from_user = [age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries]
        # Return the data
        return data_from_user
        # return 'backend received: Age:' + age + ' Diabetes:' + diabetes +' Blood Presure Problem:' + blood_pressure + ' Any transplants:' + any_transplants + ' Any chronic diseases: ' + chronic_diseases + ' Height: ' + height + ' Weight: ' + weight + ' Known allergies:' + known_allergies + ' History of canser in family:' + history_of_cancer + ' Major surgeries:' + major_surgeries
       
    # Calculate the insurance premium using the Random Forest model
    def calculate_premium(self, data):
        #result = self.model.predict([data])
        # Result is going to be a random number for now
        # Random number is generated for testing purposes
        result = random.randint(0, 10000)

        return result