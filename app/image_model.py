# import des librairies
import tensorflow as tf


# fonction de chargement du modèle
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# fonction de prédiction
def predict(model, image):
    try:
        prediction = model.predict(image)
        return prediction.tolist()
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")