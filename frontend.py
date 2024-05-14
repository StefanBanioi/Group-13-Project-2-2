import tkinter as tk
from adapter import Adapter

# Frontend class
class Frontend:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.root = tk.Tk()
        self.root.geometry("600x600")  # Increased window size
        self.root.title("Predicting Insurance Premium for Individuals (One Year)")

        # Title Label with increased font size
        self.title_label = tk.Label(self.root, text="Predicting Insurance Premium for Individuals (One Year)", font=("Helvetica", 16, "bold"))
        self.title_label.grid(row=0, columnspan=2, padx=15, pady=10)

        # Creating frames for better organization
        self.frame_labels = tk.Frame(self.root)
        self.frame_labels.grid(row=1, column=0, padx=15, pady=10)

        self.frame_entries = tk.Frame(self.root)
        self.frame_entries.grid(row=1, column=1, padx=15, pady=10)

        # Labels with increased font size
        labels = [
            "Age:", "Diabetes (Yes/No):", "Blood Pressure Problems (Yes/No):",
            "Any Transplants (Yes/No):", "Chronic Diseases (Yes/No):",
            "Height (cm):", "Weight (kg):", "Known Allergies (Yes/No):",
            "History of Cancer in Family (Yes/No):", "Major Surgeries (Yes/No):"
        ]

        for i, text in enumerate(labels):
            label = tk.Label(self.frame_labels, text=text, font=("Helvetica", 12))
            label.grid(row=i, column=0, sticky="w", pady=5)

        # Entries with increased size
        self.entries = []
        for i in range(len(labels)):
            entry = tk.Entry(self.frame_entries, width=20, font=("Helvetica", 12))
            entry.grid(row=i, column=0, pady=5)
            self.entries.append(entry)

        self.button_send = tk.Button(self.root, text="Send", command=self.send_data, font=("Helvetica", 15))
        self.button_send.grid(row=2, column=0, pady=10)

        self.button_calc = tk.Button(self.root, text="Calculate", command=self.calculate_data, font=("Helvetica", 15))
        self.button_calc.grid(row=2, columnspan=2, pady=15)

        # Result Text Widget with increased font size
        self.result_text = tk.Text(self.root, height=2, width=30, font=("Helvetica", 12), state="disabled")
        self.result_text.grid(row=3, columnspan=2, pady=10)
        



    def send_data(self):
        # Get the data from the entry widget and send it to the adapter

        age = self.entries[0].get()
        diabetes = self.entries[1].get()
        blood_pressure = self.entries[2].get()
        any_transplants = self.entries[3].get()
        chronic_diseases = self.entries[4].get()
        height = self.entries[5].get()
        weight = self.entries[6].get()
        known_allergies = self.entries[7].get()
        history_of_cancer = self.entries[8].get()
        major_surgeries = self.entries[9].get()

        response = self.adapter.request(age, diabetes, blood_pressure, any_transplants, chronic_diseases, height, weight, known_allergies, history_of_cancer, major_surgeries)
        print(response)

    def calculate_data(self):

        result = self.adapter.calculate_data([1, 1, 1, 1, 1, 170, 70, 1, 1, 1])
                
        # Clear the text box
        self.result_text.config(state="normal")
        self.result_text.delete('1.0', tk.END)


        # Insert the new value
        self.result_text.insert(tk.END, str(result))

         # Disable the text box so it's read-only
        self.result_text.config(state="disabled")
        print('test')
        
        return result

    def run(self):
        self.root.mainloop()