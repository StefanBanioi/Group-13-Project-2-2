import tkinter as tk
from adapter import Adapter

# Frontend class
class Frontend:
    def __init__(self, adapter: Adapter):
        self.adapter = adapter
        self.root = tk.Tk()
        self.root.geometry("500x500")
        self.root.title("Insurance Price Calculator")

        self.label = tk.Label(self.root, text="Enter your age:")
        self.label.grid(row=0, column=0)
        self.entry = tk.Entry(self.root)
        self.entry.grid(row=0, column=1)

        self.label1 = tk.Label(self.root, text="Enter your sex:")
        self.label1.grid(row=1, column=0)
        self.entry1 = tk.Entry(self.root)
        self.entry1.grid(row=1, column=1)

        self.label2 = tk.Label(self.root, text="Enter your location:")
        self.label2.grid(row=2, column=0)
        self.entry2 = tk.Entry(self.root)
        self.entry2.grid(row=2, column=1)

        self.button = tk.Button(self.root, text="Send", command=self.send_data)
        self.button.grid(row=3, column=0)

        self.calc_button = tk.Button(self.root, text="Calculate", command=self.calculate_data)
        self.calc_button.grid(row=3, column=1)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.grid(row=4, column=0, columnspan=2)



    def send_data(self):
        # Get the data from the entry widget and send it to the adapter

        age = self.entry.get()
        sex = self.entry1.get()
        location = self.entry2.get()
        response = self.adapter.request(age, sex, location)
        print(response)

    def calculate_data(self):
        data = self.entry.get()
        self.entry2 = tk.Entry(self.root)
        self.entry2.pack()
        result = self.adapter.perform_calculation(data)
        self.result_label.config(text=f"Result: {result}")

    def run(self):
        self.root.mainloop()