from app.computervision.registry import get_prediction_models, prediction_model
from app.computervision.models import PredictionModel, VIOLENCE_CV_MODEL_PATH

DUMMY_MODEL_PATH = "dummy_model_path"

@prediction_model(DUMMY_MODEL_PATH)
class DummyPredictionModel(PredictionModel):
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, image):
        pass

    def get_model_path(self):
        return self.model_path

    def preprocess_image(self, image):
        pass


def test_get_prediction_models():
    models = get_prediction_models()
    assert models[0].get_model_path() == VIOLENCE_CV_MODEL_PATH
    assert models[1].get_model_path() == DUMMY_MODEL_PATH
