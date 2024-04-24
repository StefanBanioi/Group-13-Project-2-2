from backend import Backend

# Adapter class
class Adapter:
    def __init__(self, backend: Backend):
        self.backend = backend

    def request(self, age , sex, location):
        return self.backend.handle_request(age , sex, location)