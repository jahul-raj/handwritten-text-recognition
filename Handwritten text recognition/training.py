import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int64)
X = X.reshape(-1, 28, 28, 1)

# Split dataset
u_train, u_test, v_train, v_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Layers
layer_con1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
layer_pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

layer_con2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
layer_pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))  # <- fixed name

layer_flatten = tf.keras.layers.Flatten()
layer_dense1 = tf.keras.layers.Dense(128, activation='relu')      # <- removed comma
layer_dense2 = tf.keras.layers.Dense(10, activation='softmax')

# Build model
model = tf.keras.models.Sequential([
    layer_con1,
    layer_pooling1,
    layer_con2,
    layer_pooling2,   # <- fixed
    layer_flatten,
    layer_dense1,     # <- fixed
    layer_dense2
])

# Compile & train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(u_train, v_train, epochs=5, validation_data=(u_test, v_test))

# Save model
os.makedirs("train", exist_ok=True)
model.save("train/text_model.keras")

# Reload & evaluate
model = tf.keras.models.load_model("train/text_model.keras")
loss, acc = model.evaluate(u_test, v_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")
