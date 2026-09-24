import os

import albumentations as A
import pandas as pd
import torch
from torch.utils.data import Dataset

from plugins.brainhemorrhage.utils import CreateChannels, load_numpy


class DataLoader(Dataset):
    def __init__(self, config: dict, mode: str = 'train') -> None:
        self.config = config
        self.mode = mode
        csv_path = config[f'{mode}_csv_path']
        self.df = self.preprocess_df(pd.read_csv(csv_path))

        self.create_channels = CreateChannels(
            config['input_mode'],
            window_widths=config['window_widths'],
            window_centers=config['window_centers']
        )

        self.transforms_all = self.create_transform(positive_sample=True)
        self.transforms_non_pos = self.create_transform(positive_sample=False)

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['filename'] = df["patient_name"].astype(str) + '_' + df["slice_number"].astype(str) + '.npy'
        return df

    def __len__(self) -> int:
        return len(self.df)

    def create_transform(self, positive_sample: bool = True) -> A.Compose:
        transform_list = [
            A.Resize(
                height=self.config['image_size'][0],
                width=self.config['image_size'][1]
            )
        ]

        if self.mode not in ['train', 'val']:
            return A.Compose(transform_list)

        augmentations = self.config['train_augmentations'] if self.mode == 'train' else self.config['val_augmentations']

        for aug_name, aug_config in augmentations.items():
            kwargs = aug_config.get("kwargs", {})
            apply_on_pos_only = aug_config.get("args", {}).get("apply_on_pos_only", 0)

            if not apply_on_pos_only or positive_sample:
                if aug_name == 'flip':
                    transform_list.append(A.VerticalFlip(**kwargs))
                elif aug_name == 'rotate':
                    transform_list.append(A.Rotate(**kwargs))
                elif aug_name == 'shift_scale':
                    transform_list.append(A.ShiftScaleRotate(**kwargs))
                elif aug_name == 'color_jitter':
                    transform_list.append(A.ColorJitter(**kwargs))

        return A.Compose(transform_list)

    def __getitem__(self, index: int) -> dict:
        df = self.df
        filename = df.iloc[index]['filename']
        binary_label = int(df.iloc[index]["hemorrhage_binary"])

        try:
            original_ct = load_numpy(os.path.join(self.config['image_dir'], filename))
            image = self.create_channels(original_ct)
        except Exception as e:
            print(f"Skipping corrupted file {filename} due to error: {e}")
            return self.__getitem__((index + 1) % len(self))

        targets = torch.tensor([binary_label], dtype=torch.float32)

        transforms = self.transforms_all if bool(binary_label) else self.transforms_non_pos
        image = transforms(image=image)['image']
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            "input_data": image,
            "targets": targets,
            "filename": filename,
            "input_cts": original_ct
        }
