from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('potatoes.h5')

# Define the class labels
class_labels = ['Healthy', 'Early Blight', 'Late Blight']

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image
    image = load_img(file_path, target_size=(256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template('result.html', label=predicted_class, image_path=file_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
