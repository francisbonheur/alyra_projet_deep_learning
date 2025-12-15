from app.computervision.models import (
    ViolencePredictionCVModel,
    VIOLENCE_CV_MODEL_PATH
)

VIOLENCE_IMAGE_PATH = "examples/violence/bagarre3.jpg"
NON_VIOLENCE_IMAGE_PATH = "examples/non_violence/film.jpg"
model = ViolencePredictionCVModel(VIOLENCE_CV_MODEL_PATH)


def test_predict_violence():
    try:
        with open(VIOLENCE_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()

        prediction = model.predict(image_bytes)
        assert prediction[0][0] == 0.7545508742332458

    except FileNotFoundError:
        print(f"Error: The file '{VIOLENCE_IMAGE_PATH}' was not found.")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None


def test_predict_non_violence():
    try:
        with open(NON_VIOLENCE_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()

        prediction = model.predict(image_bytes)
        assert prediction[0][0] == 0.0015360848046839237

    except FileNotFoundError:
        print(f"Error: The file '{NON_VIOLENCE_IMAGE_PATH}' was not found.")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None
