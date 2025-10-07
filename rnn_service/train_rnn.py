# rnn_service/train_rnn.py
from pathlib import Path
import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = Path(__file__).parent
DATA = BASE / "field_notes_database.csv"
TOKENIZER = BASE / "tokenizer_rnn.pkl"
MODEL = BASE / "model_rnn.h5"

def main():
    assert DATA.exists(), f"Crie o dataset via API /log primeiro: {DATA}"
    df = pd.read_csv(DATA)
    texts = df["texto"].astype(str).tolist()
    y = df["urgente"].astype(int).values

    tok = Tokenizer(num_words=4000, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    X = pad_sequences(tok.texts_to_sequences(texts), maxlen=40)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=4000, output_dim=32, input_length=X_tr.shape[1]),
        keras.layers.LSTM(32),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=10, batch_size=16)

    model.save(MODEL)
    with open(TOKENIZER,"wb") as f: pickle.dump(tok,f)
    print("✅ Salvos:", MODEL, TOKENIZER)

if __name__ == "__main__":
    main()
