import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def preprocess_supervised(transactions_path, labels_path):
    transactions = pd.read_csv(transactions_path)
    with open(labels_path, 'r') as f:
        labels = json.load(f)['target']
    labels_df = pd.DataFrame(list(labels.items()), columns=['id', 'Status'])
    labels_df['id'] = pd.to_numeric(labels_df['id'])
    merged = pd.merge(transactions, labels_df, on='id', how='inner')
    merged['amount'] = merged['amount'].replace(r'[^\d.]', '', regex=True).astype(float)
    merged['date'] = pd.to_datetime(merged['date'])
    merged['merchant_state'].fillna(merged['merchant_state'].mode()[0], inplace=True)
    merged['zip'].fillna(merged['zip'].mode()[0], inplace=True)
    merged['errors'].fillna(merged['errors'].mode()[0], inplace=True)
    categorical = ['use_chip', 'merchant_city', 'merchant_state', 'errors']
    le = LabelEncoder()
    for col in categorical:
        merged[col] = le.fit_transform(merged[col].astype(str))
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month
    merged['day'] = merged['date'].dt.day
    merged['hour'] = merged['date'].dt.hour
    merged['weekday'] = merged['date'].dt.weekday
    merged['Status'] = merged['Status'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = merged.drop(['Status', 'date'], axis=1)
    y = merged['Status']
    return X, y

def train_rf(transactions_path, labels_path):
    X, y = preprocess_supervised(transactions_path, labels_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_rf("data/transactions_data.csv", "data/train_fraud_labels.json")
