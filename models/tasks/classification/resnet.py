import timm
import torch.nn as nn

from models.basemodel import Model


class ResNet18(Model):
    def __init__(self,
                 optimizer,
                 criterion,
                 config,
                 init_lr=0.001,
                 metrics=[],
                 model_name='resnet18'):
        num_classes = 1

        pretrained = bool(config['pretrained'])
        self.model_name = model_name
        model = timm.create_model(model_name, pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.blocks = [model.layer1, model.layer2, model.layer3, model.layer4]

        frozen_layers_ratio = config['frozen_layers_ratio']
        if frozen_layers_ratio > 0.0:
            param_modules = []
            seen = set()

            classifier_modules = set()
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                classifier_modules = set(classifier.modules())

            for module in model.modules():
                if module in classifier_modules:
                    continue
                if any(p.requires_grad for p in module.parameters(recurse=False)) and module not in seen:
                    param_modules.append(module)
                    seen.add(module)

            num_to_freeze = int(len(param_modules) * frozen_layers_ratio)

            for module in param_modules[:num_to_freeze]:
                for param in module.parameters():
                    param.requires_grad = False

            print(f'Frozen {num_to_freeze} parameter-containing layers (excluding classifier), '
                  f'{len(param_modules) - num_to_freeze + len(classifier_modules)} remain trainable.')

            del param_modules
            del seen
            del classifier_modules

        super(ResNet18, self).__init__(model=model,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       init_lr=init_lr,
                                       metrics=metrics,
                                       config=config)
