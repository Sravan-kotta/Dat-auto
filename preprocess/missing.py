import pandas as pd
import sys  # Import sys to access command-line arguments

def clean_csv(csv_path):
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')  # or 'latin1' or 'cp1252'
    
    # Drop rows with any missing values
    df.dropna(inplace=True)
    
    # Drop duplicate rows, keeping the first occurrence
    df.drop_duplicates(inplace=True)
    
    # Save the cleaned DataFrame back to the CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:  # Check if a command-line argument is provided
        csv_file_path = sys.argv[1]  # Get the CSV file path from the command-line argument
        clean_csv(csv_file_path)  # Call the function with the CSV path
    else:
        print("Please provide the path to the CSV file.")
