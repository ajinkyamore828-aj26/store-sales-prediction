import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Dataset
data = {
    "Advertising_Cost": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    "Store_Size": [1000, 1200, 1500, 1700, 2000,
                   2200, 2500, 2700, 3000, 3200],
    "Customers": [100, 120, 150, 170, 200,
                  230, 250, 280, 300, 330],
    "Sales": [200, 250, 300, 350, 400,
              450, 500, 550, 600, 650]
}

df = pd.DataFrame(data)

X = df[["Advertising_Cost", "Store_Size", "Customers"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "sales_model.pkl")

print("Model Saved Successfully!")