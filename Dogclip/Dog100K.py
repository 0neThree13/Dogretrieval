#数据加载
import os
import json
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
import torch
from torchvision import transforms

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 76

class ImgDataset(Dataset):
    def __init__(self, root: str, subset: str, img_dataset: str):
        """
        Args:
            root (str): Root directory of the dataset.
            subset (str): Subset of the dataset ('train', 'val', 'test').
            img_dataset (str): Directory of the audio dataset.
        """
        self.root = os.path.expanduser(root)
        self.subset = subset
        self.img_dataset = img_dataset

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Dog100K(Dataset):
    text_col = 'caption'
    file_col = 'filename'
    meta = {
        'filename': 'captions.csv',  # 相对 dataset 路径
    }

    def __init__(self, root, subset='train', img_dataset='images', img_augment: nn.Module = None,text_augment: nn.Module = None,
                  seed: int = 1):

        super().__init__()
        self.root = root
        self.subset = subset
        self.img_dir = img_dataset
        self.img_augment = img_augment
        self.text_augment_fn = text_augment
        self.seed = seed
        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),  # ← 统一图像尺寸！
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.captions, self.img_paths = [], []
        self._load_meta()
        self._split_data()
        self._save_to_json()

    def _load_meta(self):
        """从 CSV 加载元数据"""
        print('Loading data...')
        path = os.path.join(self.root,self.meta['filename'])
        self.df = pd.read_csv(path)
        if self.text_col not in self.df.columns or self.file_col not in self.df.columns:
            raise ValueError(f"CSV 文件必须包含 '{self.text_col}' 和 '{self.file_col}' 两列")
        self.all_captions = self.df[self.text_col].unique()

    def _split_data(self):
        print('Splitting data...')
        if self.subset not in ['train', 'test']:
            raise ValueError("Subset must be one of 'train' or 'test'")

        np.random.seed(self.seed)

        # 打乱所有行
        df_shuffled = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # 划分训练集和测试集（默认 9:1）
        total_size = len(df_shuffled)
        train_end = int(0.9 * total_size)

        if self.subset == 'train':
            subset_df = df_shuffled.iloc[:train_end]
        elif self.subset == 'test':
            subset_df = df_shuffled.iloc[train_end:]
        else:
            raise ValueError(f"Invalid subset type: {self.subset}")

        # 遍历子集
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f'Loading {self.subset} data'):
            file_path = os.path.join(self.root, self.img_dir, row[self.file_col])
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found: {file_path}")
                continue
            self.img_paths.append(file_path)
            self.captions.append(row[self.text_col])

        # 遍历子集
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f'Loading {self.subset} data'):
            file_path = os.path.join(self.root, self.img_dir, row[self.file_col])
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found: {file_path}")
                continue
            self.img_paths.append(file_path)
            self.captions.append(row[self.text_col])


    def _save_to_json(self):
        """可选：保存为 JSON 文件以加快以后加载"""
        print('Saving data...')
        data = {
            "img_paths": self.img_paths,
            "captions": self.captions,
        }
        json_path = os.path.join(self.root, f'{self.subset}_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def __getitem__(self, index):
        """返回图像和对应的 caption"""
        image_path = self.img_paths[index]
        caption = self.captions[index]

        # image is a Tensor after this
        image = Image.open(image_path).convert("RGB")
        image = self.to_tensor(image)

        # Tokenizer 编码
        # caption = tokenizer.encode(caption)#tokenizer
        # caption = torch.tensor(caption)
        # 截断太长的 caption
        if len(caption) > MAX_LEN:
            caption = caption[:MAX_LEN]

        return image, caption


    def __len__(self):
        return len(self.img_paths)


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(captions)