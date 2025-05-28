# Credit-Card-Fraud-Detection-Using-Time-Series-Data

This project explores both **unsupervised** and **supervised** learning techniques to detect fraudulent transactions using real-world-style transactional data.

## 📁 Project Structure

```
fraud-detection-project/
├── data/                          # Data files (excluded from GitHub using .gitignore)
│   ├── transactions_data.csv
│   └── train_fraud_labels.json
├── scripts/                       # Core scripts for processing and training
│   ├── preprocess.py              # Data cleaning and feature engineering
│   ├── train_unsupervised.py      # Isolation Forest for anomaly detection
│   ├── train_supervised.py        # Random Forest for classification
│   └── evaluate.py                # Evaluation utilities and visualizations
├── main.py                        # Entry point for running models
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## ⚙ Technologies Used

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- IsolationForest (unsupervised)
- RandomForestClassifier (supervised)

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/neelapalasravani/fraud-detection-project.git
   cd fraud-detection-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your datasets in the `data/` folder:
   - `transactions_data.csv`
   - `train_fraud_labels.json`

4. Run the project:
   ```bash
   python main.py
   ```

##  Evaluation Metrics (Supervised)

- **Accuracy**: 99.96%
- **Precision**: 98.36%
- **Recall**: 77.47%
- **F1-Score**: 86.67%

##  Future Work

- Incorporate LSTM-based time series prediction for fraud pattern evolution.
- Use ensemble models to combine unsupervised + supervised outputs.

---


