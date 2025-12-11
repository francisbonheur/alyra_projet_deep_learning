import numpy as np
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = "models/mobilenetv2_violence_feature_extract.keras"


class ViolencePredictionCVModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    # fonction de preprocessing des données d'entrée
    def preprocess_image(self, image):
        try:
            image = Image.open(io.BytesIO(image))
            image = image.resize((224, 224))
            image_array = np.array(image)

            # Reshape et Normalisation 
            image_final = preprocess_input(image_array.reshape(-1, 224, 224, 3).astype("float32"))

            return image_final
        except Exception as e:
            raise ValueError("Error in image preprocessing: " + str(e))

    # fonction de prédiction
    def predict(self, image):
        try:
            prediction = self.model.predict(self.preprocess_image(image))
            return prediction.tolist()
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")


def get_model():
    model = None
    try:
        # chargement du modèle au démarrage de l'Application
        model = ViolencePredictionCVModel(MODEL_PATH)

    except Exception as e:
        print(f"Error loading model: {e}")

    return model
