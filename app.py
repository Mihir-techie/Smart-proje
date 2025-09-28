from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ---------- Load Models ----------
irrigation_model = pickle.load(open("models/irrigation_model.pkl", "rb"))
water_model = pickle.load(open("models/water_quality_model.pkl", "rb"))

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template("home.html")

@app.route("/irrigation")
def irrigation_page():
    return render_template("irrigation.html")

@app.route("/water")
def water_page():
    return render_template("water.html")

# ---------- Irrigation Prediction ----------
@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        prediction = irrigation_model.predict(final_features)[0]
        result = "💧 Irrigation Needed!" if prediction == 1 else "✅ No Irrigation Needed"
    except Exception as e:
        result = f"Error: {e}"
    return render_template("irrigation.html", prediction_text=result)

# ---------- Water Quality Prediction ----------
@app.route('/predict_water', methods=['POST'])
def predict_water():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        prediction = water_model.predict(final_features)[0]
        result = "🚱 Water Not Safe" if prediction == 1 else "💧 Water Safe to Drink"
    except Exception as e:
        result = f"Error: {e}"
    return render_template("water.html", prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
