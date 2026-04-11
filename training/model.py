import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load training data
df = pd.read_csv("training_data.csv")

X = df[["Variance", "PSNR"]]
y = df["K"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved")