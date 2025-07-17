# Wallet Credit Scoring System

## 📌 Overview
This project calculates credit scores for blockchain wallet addresses using their transaction behaviors like deposits, borrowings, liquidations, and asset diversity.

## 🧠 Methodology
A rule-based scoring system is used, composed of:

- **Activity Score**: Based on total transaction volume, frequency, and recency.
- **Risk Score**: Based on liquidation frequency, high variability, and anomalies.
- **Diversity Score**: Based on asset types and action variety.

The final score is normalized to a 0–1000 scale.

## ⚙️ Architecture & Processing Flow

JSON Input → Feature Engineering → Anomaly Detection →
Score Computation → Normalization → Output Scores


1. **Load JSON** of wallet transactions.
2. **Extract features** per wallet:
   - Total transactions
   - Unique transaction days
   - Transaction volume in USD
   - Types of actions (deposit, borrow, etc.)
3. **Detect anomalies** using Isolation Forest.
4. **Score wallets** using weighted aggregation.
5. **Normalize scores** between 0–1000.

## 📂 File Structure
wallet-credit-score/
├── credit_score_calculator.py
├── user-wallet-transactions.json
├── readme.md
├── analysis.md
└── requirements.txt


## 🧪 Requirements


pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost

Install with:
```bash
pip install -r requirements.txt
python credit_score_calculator.py

---

✅ Once you've pasted and saved the file, say **"next"** and I’ll give you the full content for `analysis.md`.


