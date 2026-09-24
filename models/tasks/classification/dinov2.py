import torch
import torch.hub
import torch.nn as nn

from models.basemodel import Model


class DINOv2(Model):
    def __init__(self,
                 optimizer,
                 criterion,
                 config,
                 init_lr=0.001,
                 metrics=[],
                 model_name='dinov2_vits14'):
        num_classes = 1

        self.model_name = model_name

        model = torch.hub.load('facebookresearch/dinov2', model_name)

        if 'vits' in model_name:
            embed_dim = 384
        elif 'vitb' in model_name:
            embed_dim = 768
        elif 'vitl' in model_name:
            embed_dim = 1024
        elif 'vitg' in model_name:
            embed_dim = 1536
        else:
            embed_dim = model.norm.normalized_shape[0] if hasattr(model, 'norm') else 384

        model.head = nn.Linear(embed_dim, num_classes)

        frozen_layers_ratio = config.get('frozen_layers_ratio', 0.0)
        if frozen_layers_ratio > 0.0:
            param_modules = []
            seen = set()

            head_modules = set()
            if hasattr(model, 'head'):
                head = model.head
                head_modules = set(head.modules())

            for module in model.modules():
                if module in head_modules:
                    continue
                if any(p.requires_grad for p in module.parameters(recurse=False)) and module not in seen:
                    param_modules.append(module)
                    seen.add(module)

            num_to_freeze = int(len(param_modules) * frozen_layers_ratio)

            for module in param_modules[:num_to_freeze]:
                for param in module.parameters():
                    param.requires_grad = False

            print(f'Frozen {num_to_freeze} parameter-containing layers (excluding head), '
                  f'{len(param_modules) - num_to_freeze + len(head_modules)} remain trainable.')

            del param_modules
            del seen
            del head_modules

        super(DINOv2, self).__init__(model=model,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     init_lr=init_lr,
                                     metrics=metrics,
                                     config=config)


class DINOv2ViTS14(DINOv2):
    def __init__(self, optimizer, criterion, config, init_lr=0.001, metrics=[]):
        super(DINOv2ViTS14, self).__init__(optimizer=optimizer,
                                           criterion=criterion,
                                           config=config,
                                           init_lr=init_lr,
                                           metrics=metrics,
                                           model_name='dinov2_vits14')
