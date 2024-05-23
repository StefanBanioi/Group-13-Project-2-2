import tensorflow as tf

class ANNModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('model_ANN.h5')
    
    def predict(self, input_data):
        #original from of data 
        #age,sex,weight,bmi,hereditary_diseases,no_of_dependents,smoker,city,bloodpressure,diabetes,regular_ex,job_title,claim

        #after transforming to one hot encoding
        #age,weight,bmi,no_of_dependents,smoker,bloodpressure,diabetes,regular_ex,sex_female,sex_male,hereditary_diseases_Alzheimer,hereditary_diseases_Arthritis,hereditary_diseases_Cancer,hereditary_diseases_Diabetes,hereditary_diseases_Epilepsy,hereditary_diseases_EyeDisease,hereditary_diseases_HeartDisease,hereditary_diseases_High BP,hereditary_diseases_NoDisease,hereditary_diseases_Obesity,city_Atlanta,city_AtlanticCity,city_Bakersfield,city_Baltimore,city_Bloomington,city_Boston,city_Brimingham,city_Brookings,city_Buffalo,city_Cambridge,city_Canton,city_Carlsbad,city_Charleston,city_Charlotte,city_Chicago,city_Cincinnati,city_Cleveland,city_Columbia,city_Columbus,city_Denver,city_Escabana,city_Eureka,city_FallsCity,city_Fargo,city_Florence,city_Fresno,city_Georgia,city_GrandForks,city_Harrisburg,city_Hartford,city_Houston,city_Huntsville,city_Indianapolis,city_IowaCity,city_JeffersonCity,city_KanasCity,city_Kingman,city_Kingsport,city_Knoxville,city_LasVegas,city_Lincoln,city_LosAngeles,city_Louisville,city_Lovelock,city_Macon,city_Mandan,city_Marshall,city_Memphis,city_Mexicali,city_Miami,city_Minneapolis,city_Minot,city_Montrose,city_Nashville,city_NewOrleans,city_NewYork,city_Newport,city_Oceanside,city_Oklahoma,city_Orlando,city_Oxnard,city_PanamaCity,city_Pheonix,city_Phildelphia,city_Pittsburg,city_Portland,city_Prescott,city_Providence,city_Raleigh,city_Reno,city_Rochester,city_Salina,city_SanDeigo,city_SanFrancisco,city_SanJose,city_SanLuis,city_SantaFe,city_SantaRosa,city_SilverCity,city_Springfield,city_Stamford,city_Syracuse,city_Tampa,city_Trenton,city_Tucson,city_Warwick,city_WashingtonDC,city_Waterloo,city_Worcester,city_York,city_Youngstown,job_title_Academician,job_title_Accountant,job_title_Actor,job_title_Analyst,job_title_Architect,job_title_Beautician,job_title_Blogger,job_title_Buisnessman,job_title_CA,job_title_CEO,job_title_Chef,job_title_Clerks,job_title_Dancer,job_title_DataScientist,job_title_DefencePersonnels,job_title_Doctor,job_title_Engineer,job_title_Farmer,job_title_FashionDesigner,job_title_FilmDirector,job_title_FilmMaker,job_title_GovEmployee,job_title_HomeMakers,job_title_HouseKeeper,job_title_ITProfessional,job_title_Journalist,job_title_Labourer,job_title_Lawyer,job_title_Manager,job_title_Photographer,job_title_Police,job_title_Politician,job_title_Singer,job_title_Student,job_title_Technician,claim

        

        # Preprocess the input data if needed
        scaler = joblib.load('scaler.pkl')
        preprocessed_data = scaler.transform(input_data)
        
        # Make predictions using the loaded model
        predictions = self.model.predict(preprocessed_data)
        
        # Postprocess the predictions if needed
        postprocessed_predictions = postprocess(predictions)
        
        return postprocessed_predictions


ann_model = ANNModel(model_path)
predictions = ann_model.predict(input_data)
print(predictions)