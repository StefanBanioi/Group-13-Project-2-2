# Backend class
import random
from Random_Forest import Random_forest as rf  


class Backend:
    def handle_request(self, age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise):
        # Process the data and return a response (value of how much the user has to pay for the insurance)

        # Check if the data is valid and return an error message if it's not 
        if age.isdigit() == False:
            return 'Age must be a number'
        if sex.lower() not in ['male', 'female']:
            return 'Sex is misspelled or not a string (accepts only: Male or Female, no numbers and case non-sensitive)'
        if weight.isdigit() == False:   
            return 'Weight must be a number in kg (no letters or special characters)'   
        if bmi.isdigit() == False:   
            return 'BMI must be a number (no letters or special characters)'   
        if no_of_dependents.isdigit() == False:
            return 'No of Dependents must be a number'
        if smoker.lower() not in ['yes', 'no']:
            return 'Smoker is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if blood_pressure.lower() not in ['yes', 'no']:
            return 'Blood pressure problem is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if diabetes.lower() not in ['yes', 'no']:
            return 'Diabetes is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'
        if regular_exercise.lower() not in ['yes', 'no']:
            return 'Regular exercise is misspelled or not a string (accepts only: Yes or No, no numbers and case non-sensitive)'

        # If the data is valid, return the response
        
        # Before returning the response, convert the strings to numerical values
        # The strings are converted to 1 if the answer is yes and 0 if the answer is no
        # Before returning the response, convert the strings to numerical values
        hereditary_diseases = 0 if hereditary_diseases == 'NoDisease' else 1 if hereditary_diseases == 'Arthritis' else 2 if hereditary_diseases == 'Alzheimer' else 3 if hereditary_diseases == 'Diabetes' else 4 if hereditary_diseases == 'Cancer' else 5 if hereditary_diseases == 'EyeDisease' else 6 if hereditary_diseases == 'Obesity' else 7 if hereditary_diseases == 'High BP' else 8 if hereditary_diseases == 'HeartDisease' else 9 
        sex = 1 if sex.lower() == 'male' else 0
        smoker = 1 if smoker.lower() == 'yes' else 0
        blood_pressure = 1 if blood_pressure.lower() == 'yes' else 0
        diabetes = 1 if diabetes.lower() == 'yes' else 0
        regular_exercise = 1 if regular_exercise.lower() == 'yes' else 0
        '''
        Arthritis = 1
        Alzheimer = 2
        Diabetes = 3
        Cancer = 4
        EyeDisease = 5
        Obesity = 6
        High BP = 7
        HeartDisease = 8
        Epilepsy = 9
        '''
       
        # We also want to put everything in a list so we can use it in the model 
        data_from_user = [age, sex, weight, bmi, hereditary_diseases, no_of_dependents, smoker, blood_pressure, diabetes, regular_exercise]
        # Return the data
        return data_from_user
        # return 'backend received: Age:' + age + ' Diabetes:' + diabetes +' Blood Presure Problem:' + blood_pressure + ' Any transplants:' + any_transplants + ' Any chronic diseases: ' + chronic_diseases + ' Height: ' + height + ' Weight: ' + weight + ' Known allergies:' + known_allergies + ' History of canser in family:' + history_of_cancer + ' Major surgeries:' + major_surgeries
       
    # Calculate the insurance premium using the Random Forest model
    def calculate_premium(self, data):

        # To do: Use the Random Forest model to calculate the premium
        #        Make that the Random Forest generated 5 aproximations and then takes the average as the result of the premium and this gets returned
        #        We do this since each time RF is runned, a different result is generated so we take an average of 5 results to get a more accurate result
    
        result = rf.predict_premium(data)
        print('test2')
         
        
        
        # Result is going to be a random number for now
        # Random number is generated for testing purposes
        # result = random.randint(100, 1000)

        return result