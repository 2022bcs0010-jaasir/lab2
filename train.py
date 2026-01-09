import os
import json
import pickle
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

print("Mohamed Jaasir Subair")
print("2022BCS0010")

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(
    max_depth=9, min_samples_leaf=12, random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mse_exp04 = mean_squared_error(y_test, pred)
r2_exp04 = r2_score(y_test, pred)
print("DT-02 MSE:", mse_exp04)
print("DT-02 R2 :", r2_exp04)


with open("output/model/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {
    "MSE": mse_exp04,
    "R2": r2_exp04
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
