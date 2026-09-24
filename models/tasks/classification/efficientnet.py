import timm
import torch.nn as nn

from models.basemodel import Model


class EfficientNet(Model):
    def __init__(self,
                 optimizer,
                 criterion,
                 config,
                 init_lr=0.001,
                 metrics=[],
                 model_name='tf_efficientnetv2_b2.in1k'):
        num_classes = 1

        pretrained = bool(config['pretrained'])
        self.model_name = model_name
        model = timm.create_model(model_name, pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

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

        super(EfficientNet, self).__init__(model=model,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           init_lr=init_lr,
                                           metrics=metrics,
                                           config=config)


class EfficientNetV2B3(EfficientNet):
    def __init__(self, optimizer, criterion, config, init_lr=0.001, metrics=[]):
        super(EfficientNetV2B3, self).__init__(optimizer=optimizer,
                                               criterion=criterion,
                                               config=config,
                                               init_lr=init_lr,
                                               metrics=metrics,
                                               model_name='tf_efficientnetv2_b3.in1k')
