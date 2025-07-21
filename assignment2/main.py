import pandas as pd

# Make sure this path is correct
excel_file_path = "LB_2.xlsx" 

try:
    # This will print a list of all sheet names in the file
    xls = pd.ExcelFile(excel_file_path)
    print("Sheets found in file:", xls.sheet_names)
except FileNotFoundError:
    print(f"File not found at: {excel_file_path}")