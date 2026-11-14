import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

COLS_TO_DROP = [
    'AnnualIncome', 
    'Experience', 
    'LoanAmount', 
    'TotalAssets', 
    'BaseInterestRate'
]
