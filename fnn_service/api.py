# fnn_service/api.py
import os, numpy as np, pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
import joblib, tensorflow as tf

BASE = Path(__file__).parent
DATA = BASE / "soil_database.csv"
SCALER = BASE / "scaler_fnn.pkl"
MODEL = BASE / "model_fnn.h5"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "FNN OK. Use /log/soil_data (POST) e /predict (POST)."

@app.route("/log/soil_data", methods=["POST"])
def log_soil():
    """
    Espera JSON com:
    {
      "chuva_mm": float, "ph_solo": float, "temp_c": float,
      "nitrogenio": float, "densidade": float,
      "rendimento_alto": 0|1 (opcional)
    }
    Se 'rendimento_alto' vier ausente, calculamos uma regra simples:
    chuva > 100 => 1; caso contrário, 0.
    """
    p = request.get_json(force=True)
    if "rendimento_alto" not in p:
        p["rendimento_alto"] = 1 if float(p["chuva_mm"]) > 100 else 0

    df = pd.DataFrame([p])
    header = not DATA.exists()
    df.to_csv(DATA, mode="a", header=header, index=False)
    return jsonify(ok=True, saved=str(DATA))

@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera JSON com os 5 recursos (sem o alvo):
    {"chuva_mm":..., "ph_solo":..., "temp_c":..., "nitrogenio":..., "densidade":...}
    """
    if not MODEL.exists() or not SCALER.exists():
        return jsonify(ok=False, error="Modelo ainda não treinado."), 400

    model = tf.keras.models.load_model(MODEL)
    scaler = joblib.load(SCALER)

    p = request.get_json(force=True)
    cols = ["chuva_mm","ph_solo","temp_c","nitrogenio","densidade"]
    x = np.array([[p[c] for c in cols]], dtype=float)
    xs = scaler.transform(x)
    prob = float(model.predict(xs, verbose=0)[0][0])
    return jsonify(ok=True, prob_rendimento_alto=prob, pred=int(prob>=0.5))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)))
