import torch
from glob import glob
from PIL import Image
import os
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

class UnlabeledImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, exts=["*.jpg", "*.png", "*.jpeg", "*.webp"]):
        self.root = root
        self.files = []
        self.transform = transform
        for ext in exts:
            self.files.extend(glob(os.path.join(root, '**/*.{}'.format(ext)), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

def set_dropout(model, p):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p

def get_dataset(name_or_path, transform=None):
    if name_or_path is None or name_or_path.lower()=='cifar10':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif name_or_path.lower()=='cifar100':
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif os.path.isdir(name_or_path):
        if transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(256),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = UnlabeledImageFolder(name_or_path, transform=transform)
    else:
        # Default to CIFAR-10 if path doesn't exist
        print(f"Warning: Dataset path '{name_or_path}' not found, using CIFAR-10 instead")
        if transform is None:
            transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ])
        dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    return dataset


def get_calibration_dataset(name_or_path, num_samples=1024, transform=None):
    """
    获取用于校准的小规模数据集
    
    Args:
        name_or_path: 数据集路径或名称
        num_samples: 校准样本数量
        transform: 数据变换
        
    Returns:
        torch.utils.data.Dataset: 校准数据集
    """
    full_dataset = get_dataset(name_or_path, transform)
    
    # 创建子集
    if len(full_dataset) > num_samples:
        indices = torch.randperm(len(full_dataset))[:num_samples]
        calib_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        calib_dataset = full_dataset
    
    return calib_dataset

