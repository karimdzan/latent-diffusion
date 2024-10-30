import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self, root_dir, train=True, limit=None):
        super(FoodDataset, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.limit = limit
        self.class_to_idx = self._get_class_mapping()
        self.image_paths, self.labels = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        labels = []
        subdir = "train" if self.train else "test"
        data_dir = os.path.join(self.root_dir, subdir)
        dirlist = os.listdir(data_dir)
        for class_name in dirlist:
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    image_paths.append(img_path)
                    labels.append(self.class_to_idx[class_name])
        if self.limit:
            image_paths = image_paths[: self.limit]
            labels = labels[: self.limit]
        return image_paths, labels

    def _get_class_mapping(self):
        subdir = "train" if self.train else "test"
        data_dir = os.path.join(self.root_dir, subdir)
        return {class_name: idx for idx, class_name in enumerate(os.listdir(data_dir))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image) * 2 - 1
        return {"img": image, "label": label}
