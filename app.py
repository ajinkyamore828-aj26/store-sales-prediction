from flask import Flask, render_template, request
import pandas as pd
import joblib

# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# LOAD MODEL
model = joblib.load("sales_model.pkl")

# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template("index.html")

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    advertising_cost = float(request.form['advertising_cost'])
    store_size = float(request.form['store_size'])
    customers = float(request.form['customers'])

    # CREATE DATAFRAME
    data = pd.DataFrame(
        [[advertising_cost, store_size, customers]],
        columns=[
            "Advertising_Cost",
            "Store_Size",
            "Customers"
        ]
    )

    # PREDICT SALES
    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sales: {round(prediction, 2)}"
    )

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)