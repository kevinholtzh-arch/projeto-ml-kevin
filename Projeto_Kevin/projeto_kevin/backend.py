from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("modelo_Kevin_v3.pkl")

@app.route("/")
def home():
    return "API rodando"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    prediction = model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)