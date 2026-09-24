import os

import numpy as np
import torch

from losses import losses
from models import models
from optimizers import optimizers


def load_model(cfg):
    backbone_name = cfg['backbone']
    task = cfg['task']
    loss_args = cfg['loss_args']
    loss_name = cfg['loss']
    loss_cls, default_args = losses[task][loss_name]
    loss_args.update(default_args)
    criterion = loss_cls(**loss_args)
    model = models[task][backbone_name](optimizer=optimizers[cfg['optimizer']], criterion=criterion, config=cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = cfg.get('checkpoint_path', None)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    if hasattr(torch.serialization, 'safe_globals'):
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device} from {checkpoint_path}")
    return model
