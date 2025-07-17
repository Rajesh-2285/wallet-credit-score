import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime

def calculate_credit_scores(input_json):
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['createdAt'] = pd.to_datetime(df['createdAt'].apply(lambda x: x['$date']))

    # Feature engineering per wallet
    features = []

    for wallet, group in df.groupby('userWallet'):
        wallet_data = {'wallet': wallet}

        # Activity features
        wallet_data['total_transactions'] = len(group)
        wallet_data['unique_days'] = group['timestamp'].dt.date.nunique()
        wallet_data['tx_per_day'] = wallet_data['total_transactions'] / max(1, wallet_data['unique_days'])

        wallet_data['days_since_first_tx'] = (datetime.now() - group['timestamp'].min()).days
        wallet_data['days_since_last_tx'] = (datetime.now() - group['timestamp'].max()).days

        # Action type distribution
        action_counts = group['action'].value_counts(normalize=True)
        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
            wallet_data[f'pct_{action}'] = action_counts.get(action, 0)

        # Asset diversity
        wallet_data['unique_assets'] = group['actionData'].apply(lambda x: x['assetSymbol']).nunique()

        # USD amount stats
        usd_values = []
        for _, row in group.iterrows():
            try:
                amount = float(row['actionData']['amount'])
                price = float(row['actionData']['assetPriceUSD'])
                usd_values.append(amount * price)
            except (ValueError, TypeError, KeyError):
                continue

        if usd_values:
            wallet_data['avg_tx_size_usd'] = np.mean(usd_values)
            wallet_data['total_volume_usd'] = np.sum(usd_values)
            wallet_data['max_tx_size_usd'] = np.max(usd_values)
            wallet_data['tx_size_std'] = np.std(usd_values)
        else:
            wallet_data['avg_tx_size_usd'] = 0
            wallet_data['total_volume_usd'] = 0
            wallet_data['max_tx_size_usd'] = 0
            wallet_data['tx_size_std'] = 0

        # Time between transactions
        time_deltas = group['timestamp'].sort_values().diff().dt.total_seconds().dropna()
        if not time_deltas.empty:
            wallet_data['avg_time_between_tx'] = time_deltas.mean()
            wallet_data['time_between_tx_std'] = time_deltas.std()
        else:
            wallet_data['avg_time_between_tx'] = 0
            wallet_data['time_between_tx_std'] = 0

        features.append(wallet_data)

    # Create features DataFrame
    features_df = pd.DataFrame(features).set_index('wallet')
    features_df.fillna(0, inplace=True)

    # Scale features
    scaler = StandardScaler()
    numeric_features = features_df.select_dtypes(include=[np.number])
    scaled_features = scaler.fit_transform(numeric_features)

    # Anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    features_df['is_anomaly'] = (iso_forest.fit_predict(scaled_features) == -1).astype(int)

    # Score components

    # Activity score
    features_df['activity_score'] = (
        0.4 * np.log1p(features_df['total_volume_usd']) +
        0.3 * np.log1p(features_df['unique_days']) +
        0.3 * (1 - features_df['days_since_last_tx'] / 365)
    )

    # Risk score
    risk_ratio = features_df['tx_size_std'] / features_df['avg_tx_size_usd'].replace(0, 1)
    features_df['risk_score'] = (
        0.4 * features_df['pct_borrow'] +
        0.3 * features_df['pct_liquidationcall'] +
        0.2 * risk_ratio +
        0.1 * features_df['is_anomaly']
    )

    # Diversity score
    max_assets = features_df['unique_assets'].max() or 1
    features_df['diversity_score'] = (
        0.6 * (features_df['unique_assets'] / max_assets) +
        0.4 * (1 - features_df['pct_deposit'])
    )

    # Raw combined score
    features_df['raw_score'] = (
        0.5 * features_df['activity_score'] +
        0.3 * (1 - features_df['risk_score']) +
        0.2 * features_df['diversity_score']
    )

    # Normalize to 0–1000 scale
    min_score = features_df['raw_score'].min()
    max_score = features_df['raw_score'].max()
    features_df['credit_score'] = ((features_df['raw_score'] - min_score) / (max_score - min_score)) * 1000
    features_df['credit_score'] = features_df['credit_score'].clip(0, 1000).astype(int)

    return features_df[['credit_score']].sort_values('credit_score', ascending=False)

# Example usage
if __name__ == "__main__":
    credit_scores = calculate_credit_scores('user-wallet-transactions.json')
    print(credit_scores)
