# main.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import shutil
import os
import subprocess



class DSMGMT():
    
    def __init__(self, master):
        self.master = master
        self.master.title("Dataset Processing and Training Software")
        self.create_widgets()

    def create_widgets(self):
        ctk.CTkLabel(self.master, text="Select CSV file:").pack(pady=10)

        # Browse Button
        self.browse_button_frame = ctk.CTkFrame(self.master)
        self.browse_button_frame.pack(pady=10)
        ctk.CTkButton(self.browse_button_frame, text="Browse", command=self.browse_file).pack()

        # Frame for Clean Data and Already Cleaned buttons
        self.clean_data_frame = ctk.CTkFrame(self.master)
        self.clean_data_frame.pack(pady=20)

        # Clean Data Button
        self.clean_data_button = ctk.CTkButton(self.clean_data_frame, text="Clean Data", command=self.clean_data)
        self.clean_data_button.pack(side='left', expand=True, fill='x')

        # Already Cleaned Button - Make it half the size by controlling the frame it's in
        self.already_cleaned_frame = ctk.CTkFrame(self.clean_data_frame, width=100)
        self.already_cleaned_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the button
        self.already_cleaned_frame.pack(side='left', padx=10)
        ctk.CTkButton(self.already_cleaned_frame, text="Already Cleaned?", command=self.show_ml_button).pack(fill='both', expand=True)

        # ML Function Button - Initially not visible
        self.ml_button_frame = ctk.CTkFrame(self.master)
        self.ml_button_frame.pack(pady=20)
        self.ml_button = ctk.CTkButton(self.ml_button_frame, text="Run ML Function", command=self.run_ml_function)
        self.ml_button.pack_forget()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.copy_file(file_path)

    def show_ml_button(self):
        # This function makes the ML button visible
        self.ml_button.pack(expand=True, fill='x')
    
    def copy_file(self, file_path):
        # Create a working directory
        working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working_directory")
        os.makedirs(working_directory, exist_ok=True)

        # Copy the CSV file to the working directory
        destination_path = os.path.join(working_directory, "input_data.csv")
        shutil.copy(file_path, destination_path)

        print(f"File copied to: {destination_path}")

    def clean_data(self):
        # Path to the copied CSV file in the working directory
        working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working_directory")
        csv_file_path = os.path.join(working_directory, "input_data.csv")
        
        # Adjusted path to the missing.py script, assuming prepro_func is outside the working_directory and at the same level as Frontend
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This goes up two levels from the current file
        missing_script_path = os.path.join(base_dir, "prepro_func", "missing.py")
        
        # Call the missing.py script with the path to the copied CSV file
        subprocess.run(["python", missing_script_path, csv_file_path], check = True)

        self.show_ml_button()

        #print to verify process
        print("Data has been cleaned")

    def run_ml_function(self):
        # Path to the ML script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This goes up two levels from the current file
        ml_script_path = os.path.join(base_dir, "ML_func", "common.py")
        
        # Assuming you need to pass the path to the cleaned CSV file to the ML script
        working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working_directory")
        csv_file_path = os.path.join(working_directory, "input_data.csv")
        
        # Call the ML script with the path to the cleaned CSV file
        subprocess.run(["python", ml_script_path, csv_file_path], check=True)
        
        print("ML function has been executed")

if __name__ == "__main__":
    # Replace CustomTk() with the appropriate class provided by customtkinter
    ctk.set_appearance_mode("Dark")
    app = ctk.CTk()
    app.geometry("640x480")
    ds_mgmt = DSMGMT(app)
    app.mainloop()
