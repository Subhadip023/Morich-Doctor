import os
import logging
import socket
from flask import Flask, request, render_template, redirect, url_for
import qrcode
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your trained models
classification_model = load_model('models/inception_model.h5')
regression_model = joblib.load('models/adaboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')

# Preprocessing function for the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to get the local network IP address
def get_local_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect() for UDP doesn't send packets
        s.connect(('10.0.0.1', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address

# Route for the home page
@app.route('/')
def index():
    ip_address = get_local_ip_address()
    print(get_local_ip_address())

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(ip_address)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")

    # Delete previous QR code image if exists
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith('ip_qr_'):
            file_path_to_delete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.remove(file_path_to_delete)

    # Save QR code image with a unique name
    qr_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'ip_qr_{ip_address}.png')
    qr_img.save(qr_img_path)

    return render_template('index.html', qr_img_path=qr_img_path)

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.info("No file part in the request")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        logging.info("No selected file")
        return redirect(url_for('index'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        # Delete previous images if they exist
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename != file.filename and not filename.startswith('ip_qr_'):
                file_path_to_delete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.remove(file_path_to_delete)
        
        file.save(file_path)
        
        # Preprocess the image for classification
        img_array = preprocess_image(file_path)
        
        # Classify the image
        logging.info("Classifying the image")
        predictions = classification_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        # Map the predicted class index to class name
        class_indices = {'sample1': 0, 'sample2': 1, 'sample3': 2, 'sample4': 3}
        class_names = list(class_indices.keys())
        classification_result = class_names[predicted_class[0]]
        logging.info(f"Classification result: {classification_result}")

        # Preprocess the image for regression
        flat_img_array = img_array.flatten().reshape(1, -1)
        scaled_img_array = scaler.transform(flat_img_array)
        pca_img_array = pca.transform(scaled_img_array)
        
        # Predict the PIP(ppm) value
        logging.info("Predicting PIP(ppm) value")
        pip_prediction = regression_model.predict(pca_img_array)[0]
        logging.info(f"PIP(ppm) prediction: {pip_prediction}")

        return render_template('index.html', classification_result=classification_result, pip_prediction=pip_prediction.round(4), image_url=file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
