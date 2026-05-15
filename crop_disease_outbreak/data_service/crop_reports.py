import os
import tensorflow as tf
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download
HF_REPO_ID = os.getenv('HF_REPO_ID')

class CropDiseasePredictor:
    def __init__(self):
        self.ModelPathPotato = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="Potato_Crop_Disease_Detection.keras"
        )

        self.ModelPathCotton = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="Cotton_Crop_Disease_Detection.keras"
        )

        self.ModelPathAll = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="Multiple_Crop_Disease_Detection.keras"
        )
        # self.IMAGE_SIZE = tuple(map(int, os.getenv('IMAGE_SIZE').split(',')))

        # Class label mappings
        self.CLASS_NAMES_POTATO = ['Early blight', 'Late blight', 'Healthy']
        self.CLASS_NAMES_COTTON = ['Aphids', 'Army Worm', 'Bacterial Blight', 'Healthy', 'Powdery Mildew', 'Target Spot']
        self.CLASS_NAMES_ALL = [
            ('Apple', 'Apple_scab'), ('Apple', 'Black_rot'), ('Apple', 'Cedar_apple_rust'), ('Apple', 'healthy'),
            ('Blueberry', 'healthy'),
            ('Cherry_(including_sour)', 'Powdery_mildew'), ('Cherry_(including_sour)', 'healthy'),
            ('Corn_(maize)', 'Cercospora_leaf_spot Gray_leaf_spot'), ('Corn_(maize)', 'Common_rust'),
            ('Corn_(maize)', 'Northern_Leaf_Blight'), ('Corn_(maize)', 'healthy'),
            ('Grape', 'Black_rot'), ('Grape', 'Esca_(Black_Measles)'),
            ('Grape', 'Leaf_blight_(Isariopsis_Leaf_Spot)'), ('Grape', 'healthy'),
            ('Orange', 'Haunglongbing_(Citrus_greening)'),
            ('Peach', 'Bacterial_spot'), ('Peach', 'healthy'),
            ('Pepper,_bell', 'Bacterial_spot'), ('Pepper,_bell', 'healthy'),
            ('Potato', 'Early_blight'), ('Potato', 'Late_blight'), ('Potato', 'healthy'),
            ('Raspberry', 'healthy'),
            ('Soybean', 'healthy'),
            ('Squash', 'Powdery_mildew'),
            ('Strawberry', 'Leaf_scorch'), ('Strawberry', 'healthy'),
            ('Tomato', 'Bacterial_spot'), ('Tomato', 'Early_blight'), ('Tomato', 'Late_blight'),
            ('Tomato', 'Leaf_Mold'), ('Tomato', 'Septoria_leaf_spot'),
            ('Tomato', 'Spider_mites Two-spotted_spider_mite'), ('Tomato', 'Target_Spot'),
            ('Tomato', 'Tomato_mosaic_virus'), ('Tomato', 'Tomato_Yellow_Leaf_Curl_Virus'),
            ('Tomato', 'healthy')
        ]

        self.__load_models()
        self.predict_crop_disease = self.predict_crop_disease

    def __load_models(self):
        self.model_potato = tf.keras.models.load_model(self.ModelPathPotato)
        self.model_cotton = tf.keras.models.load_model(self.ModelPathCotton)
        self.model_all = tf.keras.models.load_model(self.ModelPathAll)
        print("Potato input:", self.model_potato.input_shape)
        print("Cotton input:", self.model_cotton.input_shape)
        print("All input:", self.model_all.input_shape)
        print("Models loaded successfully.")


    def preprocess_image(self, image_path, model):
        """Resize image dynamically using model input shape."""

        _, height, width, channels = model.input_shape

        print(f"Using model image size: {width}x{height}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((width, height))

            img_array = np.array(img, dtype=np.float32) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        print("Processed image shape:", img_array.shape)

        return img_array

    def predict_crop_disease(self, image_path, model_type):

        if model_type == 'all':
            model = self.model_all
            class_labels = self.CLASS_NAMES_ALL

        elif model_type == 'potato':
            model = self.model_potato
            class_labels = self.CLASS_NAMES_POTATO

        elif model_type == 'cotton':
            model = self.model_cotton
            class_labels = self.CLASS_NAMES_COTTON

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Dynamic preprocessing
        img_array = self.preprocess_image(image_path, model)

        predictions = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(predictions[0][predicted_index])

        if model_type == 'all':
            crop_name, disease = class_labels[predicted_index]
        else:
            crop_name = model_type.capitalize()
            disease = class_labels[predicted_index]

        return {
            "crop": crop_name,
            "disease": disease,
            "confidence": round(confidence, 4)
        }