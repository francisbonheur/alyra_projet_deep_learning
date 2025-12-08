# import des librairies
import numpy as np
from PIL import Image
import io


# fonction de preprocessing des données d'entrée
def preprocess_image(image):
    try:

        # Convertir l'image en tableau numpy
        image = Image.open(io.BytesIO(image))

        # Redimensionner l'image à la taille attendue par le modèle
        image = image.resize((224, 224))  # redimensionnement similaire aux modele

        image_array = np.array(image)

        # Reshape et Normalisation 
        image_final = image_array.reshape(-1, 224, 224, 3).astype("float32") / 255

        return image_final
    except Exception as e:
        raise ValueError("Error in image preprocessing: " + str(e))
