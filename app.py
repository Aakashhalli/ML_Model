from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
MODEL_PATH = 'leaf_classifier_model.h5'
model = load_model(MODEL_PATH)

# Define constants
IMG_SIZE = 224
CLASS_NAMES = ['Seethapala', 'Thumbe', 'Seethaashoka', 'Tomato', 'Tamarind', 'Tecoma', 'Taro', 'Turmeric', 'Tulsi', 'Spinach1', 'Papaya', 'Pea', 'Palak(Spinach)', 'Sapota', 'Pumpkin', 'Parijatha', 'Raddish', 'Pomoegranate', 'Rose', 'Sampige', 'Nelavembu', 'Onion', 'Malabar_Spinach', 'Padri', 'Marigold', 'Nerale', 'Nooni', 'Mango', 'Neem', 'Mint', 'Jasmine', 'Lemon', 'Kasambruga', 'Jackfruit', 'Kamakasturi', 'Kohlrabi', 'Kambajala', 'Lantana', 'Malabar_Nut', 'Kepala', 'Guava', 'Globe Amarnath', 'Ganigale', 'Gasagase', 'Insulin', 'Honge', 'Hibiscus', 'Henna', 'Ganike', 'Ginger', 'Common rue(naagdalli)', 'Eucalyptus', 'Ekka', 'Citron lime (herelikai)', 'Chakte', 'Curry', 'Coriender', 'Doddpathre', 'Coffee', 'Chilly', 'Beans', 'Bringaraja', 'Bamboo', 'Camphor', 'Bhrami', 'Castor', 'Balloon_Vine', 'Catharanthus', 'Betel', 'Caricature', 'Amruthaballi', 'Aloevera', 'Badipala', 'Astma_weed', 'Arali', 'Amla', 'Ashoka']
# CLASS_NAMES = ['Class1', 'Class2', ..., 'Class77']  # Replace with your class names

# Initialize Flask app
app = Flask(__name__)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Preprocess the image
    processed_image = preprocess_image(file_path)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
