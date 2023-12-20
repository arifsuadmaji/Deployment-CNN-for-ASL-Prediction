from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model
model = load_model("model_cnn.h5")  # Sesuaikan nama model dengan nama file yang Anda simpan

# Load encoder
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Fungsi untuk preprocessing gambar
def preprocess_image(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Dapatkan file gambar dari form
        file = request.files['file']
        
        # Lakukan preprocessing pada gambar
        processed_image = preprocess_image(file)
        
        # Lakukan prediksi menggunakan model
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_label = encoder.classes_[predicted_class]
        
        return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
