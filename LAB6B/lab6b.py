import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    keras.datasets.mnist.load_data()
)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

print("Pixel values:", X_train.min(), X_train.max())
print("Example label:", y_train[0])

# Original pixel values
print("Original minimum pixel value:", X_train.min())
print("Original maximum pixel value:", X_train.max())

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Original pixel values
print("Normalized minimum pixel value:", X_train.min())
print("Normalized maximum pixel value:", X_train.max())

model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)