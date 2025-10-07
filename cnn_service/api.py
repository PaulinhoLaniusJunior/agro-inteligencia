# cnn_service/api.py
import os, base64, io, numpy as np, tensorflow as tf
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
MODEL = BASE / "model_cnn.h5"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "CNN OK. Use /log/leaf_image (POST) e /predict (POST)."

def save_b64(b64, dest: Path):
    raw = base64.b64decode(b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)

@app.route("/log/leaf_image", methods=["POST"])
def log_leaf():
    """
    Espera:
    {"label":"saudavel"|"doente", "image_b64":"data:image/png;base64,..."}
    """
    p = request.get_json(force=True)
    label = p["label"].strip().lower()
    if label not in {"saudavel","doente"}:
        return jsonify(ok=False, error="label deve ser saudavel|doente"), 400

    count = len(list((UPLOADS/label).glob("*.*")))
    dest = UPLOADS/label/f"{label}_{count:04d}.jpg"
    save_b64(p["image_b64"], dest)
    return jsonify(ok=True, saved=str(dest))

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL.exists():
        return jsonify(ok=False, error="Modelo ainda não treinado."), 400
    mdl = tf.keras.models.load_model(MODEL)

    p = request.get_json(force=True)
    raw = base64.b64decode(p["image_b64"].split(",")[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((64,64))
    arr = np.array(img)[None, ...] / 255.0
    prob = float(mdl.predict(arr, verbose=0)[0][0])
    return jsonify(ok=True, prob_doente=prob, pred=("doente" if prob>=0.5 else "saudavel"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5002)))
