# Backend class
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
        
        return 'backend received: Age:' + age + ' Diabetes:' + diabetes +' Blood Presure Problem:' + blood_pressure + ' Any transplants:' + any_transplants + ' Any chronic diseases: ' + chronic_diseases + ' Height: ' + height + ' Weight: ' + weight + ' Known allergies:' + known_allergies + ' History of canser in family:' + history_of_cancer + ' Major surgeries:' + major_surgeries
       