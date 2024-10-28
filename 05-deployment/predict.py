from flask import Flask
from flask import request
from flask import jsonify

import requests
import pickle

# Load the models
with open("dv.bin", "rb") as f:
    dv = pickle.load(f)

with open("model1.bin", "rb") as f:
    model = pickle.load(f)

app = Flask("churn")
@app.route("/predict",methods=["POST"])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    result = {
        "churn probability" : float(y_pred),
        "churn" : bool(churn)
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=9696)