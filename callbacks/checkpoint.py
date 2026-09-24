import os
import torch
from callbacks.basecallback import Callback


class Checkpoint(Callback):

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        load_best_on_end: bool = True,
    ):
        super().__init__(name="checkpoint")
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.load_best_on_end = load_best_on_end

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        try:
            self.optimizer = self.model.optimizer
        except AttributeError:
            self.optimizer = None
            print('Warning: model does not have "optimizer", only saving model states')

        self.best_score = None
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pth")

        if self.mode not in ["min", "max"]:
            raise ValueError("`mode` should be either 'min' or 'max'")

        direction = "lowest" if self.mode == "min" else "highest"
        print(f"[Checkpoint] Monitoring '{self.monitor}' to track best weights (selecting the {direction} value).")

    def load_checkpoint(self) -> None:
        if os.path.exists(self.best_checkpoint_path):
            try:
                checkpoint = torch.load(self.best_checkpoint_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                logs = checkpoint['logs']
                print(f"Checkpoint loaded from {self.best_checkpoint_path}. Starting from epoch {epoch}.")
                self.model.logs = logs
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print("No best checkpoint found. Starting from scratch.")

    def save_checkpoint(self, epoch: int, logs: dict, ckpt_name: str) -> str:
        checkpoint_path = os.path.join(self.checkpoint_dir, ckpt_name)
        try:
            data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'logs': logs
            }
            if self.optimizer is not None:
                data['optimizer_state_dict'] = self.optimizer.state_dict()
            torch.save(data, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        return checkpoint_path

    def on_train_begin(self) -> None:
        self.load_checkpoint()

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if logs is None or self.monitor not in logs:
            print(f"[Checkpoint] Warning: '{self.monitor}' not found in logs.")
            return

        current = logs[self.monitor]

        if self.best_score is None or \
           (self.mode == "min" and current < self.best_score) or \
           (self.mode == "max" and current > self.best_score):
            self.best_score = current
            self.save_checkpoint(epoch, logs, ckpt_name='best.pth')
            print(f"[Checkpoint] New best model found at epoch {epoch} with {self.monitor}: {current:.4f}")

    def on_train_end(self, logs: dict = None) -> None:
        self.save_checkpoint(0, logs if logs is not None else {}, ckpt_name='last.pth')

        if self.load_best_on_end and os.path.exists(self.best_checkpoint_path):
            print(f"[Checkpoint] Loading best checkpoint from {self.best_checkpoint_path}")
            checkpoint = torch.load(self.best_checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
