import torch
import torchvision
from .random_dataset import RandomDataset
from PIL import Image
import numpy as np

def get_dataset(dataset, data_dir, transform, train=True, download=True, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()

    elif dataset=='clothing1m':
        data_path = 'E:/BigDataset/clothing1m/'
        if train==True:
            mode="train"
        elif train==False:
            mode="test"
        elif train=='minitrain':
            mode='minitrain'
        dataset=Clothing(data_path, transform, mode)



    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)