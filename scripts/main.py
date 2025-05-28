from scripts.train_unsupervised import train_isolation_forest
from scripts.train_supervised import train_rf

if __name__ == "__main__":
    print("Running Unsupervised Learning (Isolation Forest)...")
    train_isolation_forest("data/transactions_data.csv")

    print("\nRunning Supervised Learning (Random Forest)...")
    train_rf("data/transactions_data.csv", "data/train_fraud_labels.json")
