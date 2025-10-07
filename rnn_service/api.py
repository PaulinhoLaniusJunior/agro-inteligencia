# rnn_service/api.py
import os, pickle, pandas as pd, numpy as np, tensorflow as tf
from pathlib import Path
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = Path(__file__).parent
DATA = BASE / "field_notes_database.csv"
TOKENIZER = BASE / "tokenizer_rnn.pkl"
MODEL = BASE / "model_rnn.h5"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "RNN OK. Use /log/note (POST) e /predict (POST)."

@app.route("/log/note", methods=["POST"])
def log_note():
    """
    Espera:
    {"texto":"...", "urgente":1|0}
    """
    p = request.get_json(force=True)
    df = pd.DataFrame([p])
    header = not DATA.exists()
    df.to_csv(DATA, mode="a", header=header, index=False)
    return jsonify(ok=True, saved=str(DATA))

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL.exists() or not TOKENIZER.exists():
        return jsonify(ok=False, error="Modelo ainda não treinado."), 400

    mdl = tf.keras.models.load_model(MODEL)
    tok = pickle.load(open(TOKENIZER,"rb"))
    p = request.get_json(force=True)

    seq = tok.texts_to_sequences([p["texto"]])
    X = pad_sequences(seq, maxlen=40)
    prob = float(mdl.predict(X, verbose=0)[0][0])
    return jsonify(ok=True, prob_urgente=prob, pred=int(prob>=0.5), label=("urgente" if prob>=0.5 else "rotina"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5003)))
