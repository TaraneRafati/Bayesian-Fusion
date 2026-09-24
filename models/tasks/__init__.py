from models.tasks.classification import backbones as classification_models

models = {
    'binary classification': classification_models,
}

__all__ = (
    "models",
)
