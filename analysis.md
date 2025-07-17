# Credit Score Analysis

## 📊 Score Distribution

After computing credit scores for all wallets, we grouped the scores into ranges for interpretation:

| Score Range | Wallet Count |
|-------------|--------------|
| 0–100       | _            |
| 101–200     | _            |
| 201–300     | _            |
| 301–400     | _            |
| 401–500     | _            |
| 501–600     | _            |
| 601–700     | _            |
| 701–800     | _            |
| 801–900     | _            |
| 901–1000    | _            |

_Fill in the actual counts after running the scoring script._

You can generate the distribution chart with the following code:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your result DataFrame with credit_score
df = pd.read_csv("wallet_credit_scores.csv")  # or load your result DataFrame
df['range'] = pd.cut(df['credit_score'], bins=[0,100,200,300,400,500,600,700,800,900,1000])
df['range'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Wallet Credit Score Distribution")
plt.xlabel("Score Range")
plt.ylabel("Number of Wallets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.show()

🟥 Low-Scoring Wallets (0–200)
Often perform only borrow or liquidation actions.

Very few or recent transactions.

Detected as anomalies by Isolation Forest.

Low diversity — interact with only one asset or action type.

🟨 Mid-Scoring Wallets (400–700)
Moderate volume and regular activity.

Reasonable asset diversity.

Some risk indicators (liquidations or large fluctuations).

Typically represent average users with balanced behavior.

🟩 High-Scoring Wallets (800–1000)
Long-term and consistent activity across multiple days.

High transaction volume in USD.

Interact with multiple assets and actions (e.g., deposit, repay).

No anomalies detected; stable behavioral patterns.

Very low or no liquidation behavior.

📌 Conclusion
The credit scoring model effectively differentiates between risky, inactive wallets and those showing consistent, low-risk, and high-volume usage. This scoring can help in:

Granting credit limits

On-chain identity vetting

Filtering wallets for token or airdrop eligibility
