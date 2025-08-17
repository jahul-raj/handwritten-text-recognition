import os
import cv2
import numpy as np
import tensorflow as tf
import pytesseract as ts
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__)

os.makedirs("upload", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Load trained Keras model
MODEL_PATH = "train/text_model.keras"
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join('upload', filename)
    file.save(filepath)

    # Read and preprocess
    img = cv2.imread(filepath)
    if img is None:
        return "Error: Could not read uploaded image"
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def remove_noise(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def threshold_img(image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return 255 - binary

    rImg = remove_noise(gray_img)
    upscaled = cv2.resize(rImg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    binary = threshold_img(upscaled)

    cv2.imwrite('temp/output.png', binary)

    # OCR (Tesseract)
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6'
    text = ts.image_to_string(binary, config=custom_config)

    return render_template("result.html", text=text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
