import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

print("Mohamed Jaasir Subair")
print("2022BCS0010")

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42)

model = Ridge(alpha=0.7, fit_intercept=True)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse_exp02 = mean_squared_error(y_test, pred)
r2_exp02 = r2_score(y_test, pred)

print("LR-02 (Standardized) MSE:", mse_exp02)
print("LR-02 (Standardized) R2 :", r2_exp02)

with open("output/model/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {
    "MSE": mse_exp02,
    "R2": r2_exp02
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
