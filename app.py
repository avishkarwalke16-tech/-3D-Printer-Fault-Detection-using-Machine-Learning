from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Must match column names scaler was trained on
FEATURE_NAMES = ["X-direction", "Y-direction", "Z-direction"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        x_dir = float(request.form["x_direction"])
        y_dir = float(request.form["y_direction"])
        z_dir = float(request.form["z_direction"])

        # Use DataFrame with correct feature names to avoid sklearn warnings
        input_df = pd.DataFrame([[x_dir, y_dir, z_dir]], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            result = f"⚠️ Problem Detected! ({probability:.1f}% fault probability)"
            status = "warn"
        else:
            result = f"✅ No Problem Detected ({probability:.1f}% fault probability)"
            status = "ok"

    except ValueError:
        result = "❌ Please enter valid numeric values for all three fields."
        status = "warn"
    except Exception as e:
        result = f"❌ Error: {str(e)}"
        status = "warn"

    return render_template("index.html", prediction_text=result, status=status)

if __name__ == "__main__":
    app.run(debug=True)
