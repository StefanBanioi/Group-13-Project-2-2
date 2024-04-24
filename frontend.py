import tkinter as tk
from adapter import Adapter

# Frontend class
class Frontend:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.root = tk.Tk()
        self.root.geometry("500x500")
        self.root.title("Insurance Price Calculator")
        #Enter your age
        self.label = tk.Label(self.root, text="Enter your age:")
        self.label.grid(row=0, column=0)
        self.entry = tk.Entry(self.root)
        self.entry.grid(row=0, column=1)
        #Enter your diabetes status
        self.label1 = tk.Label(self.root, text="Enter your diabetes status: (Yes/No)")
        self.label1.grid(row=1, column=0)
        self.entry1 = tk.Entry(self.root)
        self.entry1.grid(row=1, column=1)
        #Enter your blood pressure problems
        self.label2 = tk.Label(self.root, text="Enter your blood pressure problems:")
        self.label2.grid(row=2, column=0)
        self.entry2 = tk.Entry(self.root)
        self.entry2.grid(row=2, column=1)
        #Enter if you had any transplants
        self.label3 = tk.Label(self.root, text="Enter if you had any transplants:")
        self.label3.grid(row=3, column=0)
        self.entry3 = tk.Entry(self.root)
        self.entry3.grid(row=3, column=1)
        #Enter if you had any chronic diseases
        self.label4 = tk.Label(self.root, text="Enter if you had any chronic diseases:")
        self.label4.grid(row=4, column=0)
        self.entry4 = tk.Entry(self.root)   
        self.entry4.grid(row=4, column=1)
        #Enter your height in cm
        self.label5 = tk.Label(self.root, text="Enter your height in cm:")
        self.label5.grid(row=5, column=0)
        self.entry5 = tk.Entry(self.root)
        self.entry5.grid(row=5, column=1)
        #Enter your weight in kg
        self.label6 = tk.Label(self.root, text="Enter your weight in kg:")
        self.label6.grid(row=6, column=0)
        self.entry6 = tk.Entry(self.root)
        self.entry6.grid(row=6, column=1)
        #Enter known allergies
        self.label7 = tk.Label(self.root, text="Enter known allergies:")
        self.label7.grid(row=7, column=0)
        self.entry7 = tk.Entry(self.root)
        self.entry7.grid(row=7, column=1)
        #Enter History of Cancer In Family(Yes/No)
        self.label8 = tk.Label(self.root, text="Enter History of Cancer In Family(Yes/No):")
        self.label8.grid(row=8, column=0)
        self.entry8 = tk.Entry(self.root)
        self.entry8.grid(row=8, column=1)
        #Enter if you had any major surgeries
        self.label9 = tk.Label(self.root, text="Enter if you had any major surgeries:")
        self.label9.grid(row=9, column=0)
        self.entry9 = tk.Entry(self.root)
        self.entry9.grid(row=9, column=1)

        self.button = tk.Button(self.root, text="Send", command=self.send_data)
        self.button.grid(row=12, column=0)

        self.calc_button = tk.Button(self.root, text="Calculate", command=self.calculate_data)
        self.button.grid(row=13, column=0)
        self.calc_button.grid(row=13, column=1)
        self.result_label = tk.Label(self.root, text="Result: ")



    def send_data(self):
        # Get the data from the entry widget and send it to the adapter

        age = self.entry.get()
        diabetes = self.entry1.get()
        blood_pressure = self.entry2.get()
        any_transplants = self.entry3.get()
        chronic_diseases = self.entry4.get()
        height = self.entry5.get()
        weight = self.entry6.get()
        known_allergies = self.entry7.get()
        history_of_cancer = self.entry8.get()
        major_surgeries = self.entry9.get()

        response = self.adapter.request(age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries)
        print(response)

    def calculate_data(self):
        data = self.entry.get()
        self.entry2 = tk.Entry(self.root)
        self.entry2.pack()
        result = self.adapter.perform_calculation(data)
        self.result_label.config(text=f"Result: {result}")

    def run(self):
        self.root.mainloop()