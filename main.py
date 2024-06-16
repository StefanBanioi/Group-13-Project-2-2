from backend import Backend
from adapter import Adapter
from frontend import Frontend

# Create an instance of Backend
backend = Backend()

# Create an instance of Adapter
adapter = Adapter(backend)

# Create an instance of Frontend 
frontend = Frontend(adapter)

# Start the GUI
frontend.run()