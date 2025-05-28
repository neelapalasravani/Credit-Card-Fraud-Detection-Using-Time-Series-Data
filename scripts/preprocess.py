import pandas as pd

def preprocess_transactions(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].replace(r'[^\d.]', '', regex=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna()
    return df
