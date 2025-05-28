import pandas as pd
from sklearn.ensemble import IsolationForest
from scripts.preprocess import preprocess_transactions

def train_isolation_forest(csv_path):
    df = preprocess_transactions(csv_path)
    df = df.groupby('date')['amount'].sum().reset_index()
    df['amount_difference'] = df['amount'].diff()
    df.dropna(inplace=True)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['amount', 'amount_difference']])
    df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})
    print(df.head())
    return df

if __name__ == "__main__":
    train_isolation_forest("data/transactions_data.csv")
