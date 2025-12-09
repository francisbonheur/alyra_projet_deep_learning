# import des librairies
import numpy as np
from PIL import Image
import io
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# fonction de preprocessing des données d'entrée
def preprocess_image(image):
    try:
        # Convertir l'image en tableau numpy
        image = Image.open(io.BytesIO(image))

        timestamp = time.time()

        # Redimensionner l'image à la taille attendue par le modèle
        image = image.resize((224, 224))  # redimensionnement similaire aux modèles

        image.save(f"preprocessing/preprocessed_{int(timestamp)}.jpg")

        image_array = np.array(image)

        # Reshape et Normalisation 
        image_final = preprocess_input(image_array.reshape(-1, 224, 224, 3).astype("float32"))

        return image_final
    except Exception as e:
        raise ValueError("Error in image preprocessing: " + str(e))
