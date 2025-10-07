# fnn_service/train_fnn.py
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import joblib

BASE = Path(__file__).parent
DATA = BASE / "soil_database.csv"
SCALER = BASE / "scaler_fnn.pkl"
MODEL = BASE / "model_fnn.h5"

def main():
    assert DATA.exists(), f"Crie o dataset via API /log primeiro: {DATA}"

    df = pd.read_csv(DATA)
    features = ["chuva_mm","ph_solo","temp_c","nitrogenio","densidade"]
    target = "rendimento_alto"
    X = df[features].values
    y = df[target].values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr_s.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_tr_s, y_tr, validation_data=(X_va_s, y_va), epochs=25, batch_size=16)

    model.save(MODEL)
    joblib.dump(scaler, SCALER)
    print("✅ Salvos:", MODEL, SCALER)

if __name__ == "__main__":
    main()
