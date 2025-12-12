"""
models.py : Module de définition des modèles de prédiction

Contient les classes de chargement des modèles de computer vision
pour la détection de contenu inapproprié.
 - images à caractère sexuel
 - images contenant des scènes de violence

Pour chaque modèle, une nouvelle implémentation de la classe
PredictionCVModel doit être créée et une instance de cette nouvelle
classe doit être rajoutée à la liste de models renvoyés par la
méthode get_models.
"""

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


VIOLENCE_CV_MODEL_PATH = "models/mobilenetv2_violence_feature_extract.keras"


class PredictionModel(ABC):
    """
    Classe abstraite définisant les méthodes communes 
    à tous les modèles de prédiction.
    """
    @abstractmethod
    def predict(self, image):
        """
        Méthode de prédiction
        
        :param image: image à utiliser pour la prédiction
        :return: résultat de la prédiction
        """
        pass

    @abstractmethod
    def get_model_path(self):
        """
        Renvoie le chemin d'accès au modèle

        :return: chemin du modèle
        """
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """
        Méthode de pré-traitement de l'image avant la prédiction

        :param image: image à pré-traiter
        :return: image pré-traitée
        """
        pass


class ViolencePredictionCVModel(PredictionModel):
    """
    Classe de chargement du modèles de prédiction de violence
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    def get_model_path(self):
        return self.model_path
    
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


def get_models() -> list[PredictionModel]:
    """
    Fonction de chargement des modèles de prédiction
    """
    models: list[PredictionModel] = []

    try:
        models.append(ViolencePredictionCVModel(VIOLENCE_CV_MODEL_PATH))

    except Exception as e:
        print(f"Error loading models: {e}")

    return models
