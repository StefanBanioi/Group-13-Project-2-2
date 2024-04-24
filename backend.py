# Backend class
class Backend:
    def handle_request(self, age , sex, location):
        # Process the data and return a response
        # Data can be anything, such as a string, integer, or list
        # In this example, we simply return the data
        # Here the data is age, sex and location
        # So return the data from above 
        # The age can only be a number
        # The sex can only be a string and only be man or woman (case non-sensitive)
        # The location can only be a string and not include any numbers
        
        if age.isdigit() == False:
            return 'Age must be a number'
        if type(sex) != str or sex.lower() not in ['man', 'woman']:
            return 'Sex is not a string or misspelled (accepts only: Man or Woman, no numbers and case non-sensitive)'
        if location.isalpha() == False:
            return 'Location is not a string or misspelled (accepts only: City or Country and no numbers)'
        return 'backend received: ' + age + ' ' + sex + ' ' + location 

       