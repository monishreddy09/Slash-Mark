import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("artifacts/fraud_detection_model.joblib")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Create the same engineered features used during training
amount = df["Amount"].astype(float).fillna(0.0)
df["Amount_Log1p"] = np.log1p(np.abs(amount))
df["Amount_Sqrt"] = np.sqrt(np.abs(amount))
df["Amount_IsZero"] = (amount == 0).astype(int)

time_sec = df["Time"].astype(float).fillna(0.0)
day_seconds = 24 * 60 * 60

df["Time_Sin"] = np.sin(
    2 * np.pi * (time_sec % day_seconds) / day_seconds
)
df["Time_Cos"] = np.cos(
    2 * np.pi * (time_sec % day_seconds) / day_seconds
)
df["Time_Hour"] = ((time_sec // 3600) % 24).astype(float)

# Select first transaction
samples = df.drop("Class", axis=1).head(10)

predictions = model.predict(samples)
probabilities = model.predict_proba(samples)[:, 1]

for i in range(len(samples)):
    print(
        f"Transaction {i+1}: "
        f"Probability={probabilities[i]:.4f}, "
        f"Prediction={'FRAUD' if predictions[i] == 1 else 'LEGITIMATE'}"
    )

