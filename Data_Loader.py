import torch
import torchvision
import numpy as np
from torch.utils import data
from torchvision import transforms


class Batch:
    def __init__(
        self,
        imgs=None,
        labels=None,
        ind=None,
    ):
        self.imgs = imgs
        self.labels = labels
        self.ind = ind

    def to(self, device):
        self.imgs = self.imgs.to(device) if self.imgs is not None else None
        self.labels = self.labels.to(device) if self.labels is not None else None
        self.ind = self.ind.to(device) if self.ind is not None else None
        return self


def collator(items):
    imgs, labels, ind= zip(*items)
    imgs = torch.stack(imgs, dim=0)
    ind = torch.tensor(ind)
    labels = torch.tensor(labels)
    return Batch(imgs=imgs, ind=ind, labels=labels)


class CVDataset(data.Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.labels_num = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item][0], self.data[item][1], [item] * self.labels_num


class CIFAR_dataset:
    def __init__(self, train_batch_size, test_batch_size, labeled_num, resize=None):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_set, self.labeled_set, self.test_set = self.dataset_split(
            labeled_num, resize
        )
        self.train_dataset = CVDataset(self.train_set)
        self.labeled_dataset = CVDataset(self.labeled_set)
        self.test_dataset = CVDataset(self.test_set)
        self.collator = collator

    def dataset_split(self, labeled_num, resize=None):
        train_trans = [transforms.ToTensor()]
        if resize:
            train_trans.insert(0, transforms.Resize(resize))
        train_trans = transforms.Compose(train_trans)
        CIFAR_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, transform=train_trans, download=True
        )
        test_trans = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.49139968, 0.48215841, 0.44653091),
                (0.24703223, 0.24348513, 0.26158784),
            ),
        ]
        if resize:
            test_trans.insert(0, transforms.Resize(resize))
        test_trans = transforms.Compose(test_trans)
        CIFAR_test = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=test_trans, download=True
        )
        labels = np.array([item[1] for item in CIFAR_train])
        new_trainset = []
        labeled_trainset = []
        for i in range(10):
            ind = np.where(labels == i)[0]
            np.random.shuffle(ind)
            train_ind = ind[:labeled_num]
            unlabeled_ind = ind[labeled_num:]
            for idx in train_ind:
                new_trainset.append(CIFAR_train[idx])
                labeled_trainset.append(CIFAR_train[idx])
            for idx in unlabeled_ind:
                new_trainset.append((CIFAR_train[idx][0], -1))
        return new_trainset, labeled_trainset, CIFAR_test

    def train_data_loader(self):
        return data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def labeled_data_loader(self):
        return data.DataLoader(
            dataset=self.labeled_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_data_loader(self):
        return data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


class MNIST_dataset:
    def __init__(self, train_batch_size, test_batch_size, labeled_num, resize=None):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_set, self.labeled_set, self.test_set = self.dataset_split(
            labeled_num, resize
        )
        self.train_dataset = CVDataset(self.train_set)
        self.labeled_dataset = CVDataset(self.labeled_set)
        self.test_dataset = CVDataset(self.test_set)
        self.collator = collator

    def dataset_split(self, labeled_num, resize=None):
        train_trans = [transforms.ToTensor()]
        if resize:
            train_trans.insert(0, transforms.Resize(resize))
        train_trans = transforms.Compose(train_trans)
        MNIST_train = torchvision.datasets.MNIST(
            root="./data", train=True, transform=train_trans, download=True
        )
        test_trans = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.49139968),
                (0.24703223),
            ),
        ]
        if resize:
            test_trans.insert(0, transforms.Resize(resize))
        test_trans = transforms.Compose(test_trans)
        MNIST_test = torchvision.datasets.MNIST(
            root="./data", train=False, transform=test_trans, download=True
        )
        labels = np.array([item[1] for item in MNIST_train])
        new_trainset = []
        labeled_trainset = []
        for i in range(10):
            ind = np.where(labels == i)[0]
            np.random.shuffle(ind)
            train_ind = ind[:labeled_num]
            unlabeled_ind = ind[labeled_num:]
            for idx in train_ind:
                new_trainset.append(MNIST_train[idx])
                labeled_trainset.append(MNIST_train[idx])
            for idx in unlabeled_ind:
                new_trainset.append((MNIST_train[idx][0], -1))
        return new_trainset, labeled_trainset, MNIST_test

    def train_data_loader(self):
        return data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def labeled_data_loader(self):
        return data.DataLoader(
            dataset=self.labeled_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_data_loader(self):
        return data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )
    

class FashionMNIST_dataset:
    def __init__(self, train_batch_size, test_batch_size, labeled_num, resize=None):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_set, self.labeled_set, self.test_set = self.dataset_split(
            labeled_num, resize
        )
        self.train_dataset = CVDataset(self.train_set)
        self.labeled_dataset = CVDataset(self.labeled_set)
        self.test_dataset = CVDataset(self.test_set)
        self.collator = collator

    def dataset_split(self, labeled_num, resize=None):
        train_trans = [transforms.ToTensor()]
        if resize:
            train_trans.insert(0, transforms.Resize(resize))
        train_trans = transforms.Compose(train_trans)
        FashionMNIST_train = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=train_trans, download=True
        )
        test_trans = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.49139968),
                (0.24703223),
            ),
        ]
        if resize:
            test_trans.insert(0, transforms.Resize(resize))
        test_trans = transforms.Compose(test_trans)
        FashionMNIST_test = torchvision.datasets.FashionMNIST(
            root="./data", train=False, transform=test_trans, download=True
        )
        labels = np.array([item[1] for item in FashionMNIST_train])
        new_trainset = []
        labeled_trainset = []
        for i in range(10):
            ind = np.where(labels == i)[0]
            np.random.shuffle(ind)
            train_ind = ind[:labeled_num]
            unlabeled_ind = ind[labeled_num:]
            for idx in train_ind:
                new_trainset.append(FashionMNIST_train[idx])
                labeled_trainset.append(FashionMNIST_train[idx])
            for idx in unlabeled_ind:
                new_trainset.append((FashionMNIST_train[idx][0], -1))
        return new_trainset, labeled_trainset, FashionMNIST_test

    def train_data_loader(self):
        return data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def labeled_data_loader(self):
        return data.DataLoader(
            dataset=self.labeled_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_data_loader(self):
        return data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )