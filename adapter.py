from backend import Backend

# Adapter class
class Adapter:
    def __init__(self, backend: Backend):
        self.backend = backend

    def request(self, age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries):
        return self.backend.handle_request(age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries)