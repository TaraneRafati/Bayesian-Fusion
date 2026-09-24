from losses.bce import BCEWithLogitsLoss

losses = {
    'binary classification': {
        'BCEWithLogitsLoss': (BCEWithLogitsLoss, {}),
    },
}

__all__ = (
    "losses",
)
