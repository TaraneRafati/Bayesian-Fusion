import gc

import torch
from tqdm import tqdm

from metrics import Mean


class Model(torch.nn.Module):
    def __init__(self, model, optimizer, criterion, config, init_lr=0.001, metrics=[]):
        super(Model, self).__init__()
        self.model = model
        self.init_lr = init_lr
        self.optimizer = optimizer(self.model.parameters(), lr=init_lr)
        self.criterion = criterion
        self.metrics = [Mean(name='loss')] + metrics
        self.logs = {}
        self.config = config

    def preprocess_input(self, x):
        device = x.device
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to(device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to(device)
        return (x - mean[None, :, None, None]) / std[None, :, None, None]

    def forward(self, x):
        x = self.preprocess_input(x)
        return self.model(x)

    def train_step(self, data, device):
        x = data['input_data'].to(device)
        y = data['targets'].to(device)

        self.optimizer.zero_grad()
        prediction = self(x)

        loss = self.criterion(prediction, y)
        loss.backward()
        self.optimizer.step()

        self.metrics[0].update_states(loss.item())

        for metric in self.metrics[1:]:
            metric.update_states(prediction, y)

        del y, x, prediction
        gc.collect()
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data, device, prefix='val_'):
        x = data['input_data'].to(device)
        y = data['targets'].to(device)

        with torch.no_grad():
            prediction = self(x)

        loss = self.criterion(prediction, y)
        self.metrics[0].update_states(loss.item())

        for metric in self.metrics[1:]:
            metric.update_states(prediction, y)
        del y, x, prediction
        gc.collect()
        return {prefix + metric.name: metric.result() for metric in self.metrics}

    def print_logs(self, epoch, train_logs, val_logs=None, verbose=1):
        if verbose > 0:
            print(f'Epoch {epoch}')
            print(f'Train logs: {train_logs}')
            if val_logs:
                print(f'Validation logs: {val_logs}')

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def fit(self,
            train_datagen,
            epochs,
            train_step_per_epoch=None,
            valid_datagen=None,
            valid_step_per_epoch=None,
            callbacks=[],
            verbose=1,
            init_epoch=1,
            device=None):
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model.to(device)

        for callback in callbacks:
            callback.on_train_begin()

        train_step_per_epoch = len(train_datagen) if train_step_per_epoch is None else train_step_per_epoch
        valid_step_per_epoch = len(valid_datagen) if valid_step_per_epoch is None else valid_step_per_epoch

        for epoch in range(init_epoch, epochs + 1):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            current_logs = {}
            train_logs = {}
            train_pbar = tqdm(enumerate(train_datagen), desc=f'Training epoch {epoch}/{epochs}', total=train_step_per_epoch, leave=False)
            self.train()
            for step, data in train_pbar:
                for callback in callbacks:
                    callback.on_train_batch_begin(step)

                batch_logs = self.train_step(data, device)
                train_pbar.set_postfix({'loss': batch_logs['loss']})

                for callback in callbacks:
                    callback.on_train_batch_end(step, logs=batch_logs)
                if step + 1 > train_step_per_epoch:
                    break
                train_logs.update(batch_logs)
            current_logs.update(batch_logs)
            self.reset_metrics()

            val_logs = {}
            if valid_datagen is not None:
                val_pbar = tqdm(enumerate(valid_datagen), desc=f'Validating epoch {epoch}/{epochs}', total=valid_step_per_epoch, leave=False)
                self.eval()
                for step, data in val_pbar:
                    for callback in callbacks:
                        callback.on_test_batch_begin(step)
                    batch_logs = self.test_step(data, device)
                    val_pbar.set_postfix({'val_loss': batch_logs['val_loss']})

                    for callback in callbacks:
                        callback.on_test_batch_end(step, logs=batch_logs)
                    if step + 1 > valid_step_per_epoch:
                        break
                self.reset_metrics()
                current_logs.update(batch_logs)
                val_logs.update(batch_logs)

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=current_logs)

            self.print_logs(epoch, train_logs, val_logs, verbose)

            self.logs[epoch] = current_logs

        for callback in callbacks:
            callback.on_train_end(self.logs)

        return self.logs
