from plugins.brainhemorrhage.tasks.classification import DataLoader as ClassificationDataLoader

dataloaders = {
    'binary classification': ClassificationDataLoader,
}

__all__ = (
    "dataloaders",
)
