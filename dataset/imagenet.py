import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)
    


class SubCustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        # self.feature_files = [f"{i}.npy" for i in range(1281167)]
        # self.label_files = [f"{i}.npy" for i in range(1281167)]
        self.feature_files = []
        self.label_files = []
        
        for root, dirs, files in os.walk(self.feature_dir):
            for file in files:
                self.feature_files.append(os.path.join(root, file))
        
        for root, dirs, files in os.walk(self.label_dir):
            for file in files:
                self.label_files.append(os.path.join(root, file))

        self.feature_files = sorted(self.feature_files)
        self.label_files = sorted(self.label_files)

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]
        
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_file = feature_file.replace('ten_crop/', 'ten_crop_105/')
            label_file = label_file.replace('ten_crop/', 'ten_crop_105/')
        else:
            feature_file = feature_file
            label_file = label_file
            
        features = np.load(feature_file)
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(label_file)
        return torch.from_numpy(features), torch.from_numpy(labels)


class SubsetWithStartIndex(Dataset):
    def __init__(self, dataset, start_index=0, end_index=None):
        """
        dataset: 原始数据集
        start_index: 从该索引开始迭代
        end_index: 可选，迭代结束的索引，如果为None，则迭代到数据集末尾
        """
        self.dataset = dataset
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else len(dataset)

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        # 由于我们从start_index开始，我们需要调整索引
        orig_idx = idx + self.start_index
        return self.dataset[orig_idx]


# def build_imagenet(args, transform):
#     return ImageFolder(args.data_path, transform=transform)

def is_valid_file(filename):

    return not filename.endswith("n06596364_9591.JPEG")

def build_imagenet(args, transform, start_index=0):
    dataset = ImageFolder(args.data_path, transform=transform)
    return SubsetWithStartIndex(dataset, start_index)


def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return SubCustomDataset(feature_dir, label_dir)