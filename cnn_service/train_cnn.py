# cnn_service/train_cnn.py
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

BASE = Path(__file__).parent
DATA = BASE / "uploads"
MODEL = BASE / "model_cnn.h5"

def main():
    img_size = (64,64)
    batch = 16

    train_ds = keras.preprocessing.image_dataset_from_directory(
        DATA, labels="inferred", label_mode="binary", image_size=img_size,
        validation_split=0.2, subset="training", seed=42, batch_size=batch
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        DATA, labels="inferred", label_mode="binary", image_size=img_size,
        validation_split=0.2, subset="validation", seed=42, batch_size=batch
    )

    norm = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y: (norm(x), y))
    val_ds   = val_ds.map(lambda x,y: (norm(x), y))

    model = keras.Sequential([
        keras.layers.Input(shape=img_size+(3,)),
        keras.layers.Conv2D(16,3,activation="relu"), keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32,3,activation="relu"), keras.layers.MaxPooling2D(),
        keras.layers.Flatten(), keras.layers.Dense(32,activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save(MODEL)
    print("✅ Salvo:", MODEL)

if __name__ == "__main__":
    main()
