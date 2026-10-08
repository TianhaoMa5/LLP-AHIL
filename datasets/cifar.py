import os.path as osp
import pickle
import numpy as np
import scipy.io as sio

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter, OrderedDict
import h5py
from MLclf import MLclf

from datasets import tran as T
from datasets.rand import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

from datasets import transform as T1
from datasets.randaugment_grey import RandomAugment as RandomAugment1

import pickle
import os
from PIL import Image

label_map = {}
class_mapping={}


def extract_labels_from_class_dict(class_dict):
    for class_idx, image_indices in enumerate(class_dict.values()):
        for image_index in image_indices:
            label_map[image_index] = class_idx
    sorted_label_map = dict(sorted(label_map.items()))
    labels = list(sorted_label_map.values())
    return labels


def load_mini_imagenet_data(dspth, split='train'):
    if split == 'train':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-train.pkl')
    elif split == 'val':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-val.pkl')
    elif split == 'test':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-test.pkl')
    else:
        raise ValueError("Invalid split; expected 'train', 'val', or 'test'")

    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['image_data']
    labels = data_dict['class_dict']

    return data, labels


def merge_train_val_test(dspth):
    train_data, train_labels = load_mini_imagenet_data(dspth, split='train')
    val_data, val_labels = load_mini_imagenet_data(dspth, split='val')
    test_data, test_labels = load_mini_imagenet_data(dspth, split='test')

    merged_data = np.concatenate([train_data, val_data, test_data], axis=0)

    class_mapping = {}
    class_label = 0
    merged_labels = [None] * len(merged_data)
    m_labels={**test_labels, **train_labels, **val_labels}
    new_labels = {key: idx for idx, key in enumerate(m_labels.keys())}
    sample_labels = []

    # Iterate over the m_labels dictionary and assign the label to each sample
    for key, samples in m_labels.items():
        label = new_labels[key]  # Get the label for the current key
        sample_labels.extend([label] * len(samples))

    return merged_data, sample_labels



def load_tiny_imagenet_val(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    with open(os.path.join(root, 'tiny-imagenet-200/val', 'val_annotations.txt'), 'r') as f:
        for line in f:
            parts = line.split('\t')
            image_name = parts[0]
            class_name = parts[1]
            bbox = list(map(int, parts[2:]))

            if class_name not in label_map:
                label_map[class_name] = n_class
                n_class += 1

            image_path = os.path.join(root, 'tiny-imagenet-200/val', 'images', image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image = np.array(image)

            if len(image.shape) != 3 or image.shape[2] != 3:
                continue

            datalist.append(image)
            labels.append(label_map[class_name])

    return np.array(datalist), np.array(labels), n_class
def load_tiny_imagenet_data(root, image_size=(64, 64)):
    datalist = []
    labels = []
    n_class = 0

    # Loop through each class folder
    for class_folder in os.listdir(os.path.join(root, 'tiny-imagenet-200/train')):
        class_folder_path = os.path.join(root, 'tiny-imagenet-200/train', class_folder)
        if os.path.isdir(class_folder_path):
            label_map[class_folder] = n_class
            n_class += 1
            for image_file in os.listdir(os.path.join(class_folder_path, 'images')):
                image_path = os.path.join(class_folder_path, 'images', image_file)
                # Load and resize image
                image = Image.open(image_path)
                image = image.resize(image_size)
                # Convert to numpy array
                image = np.array(image)
                # Ensure image has 3 channels
                if len(image.shape) != 3 or image.shape[2] != 3:
                    continue
                # Append to data list and label list
                datalist.append(image)
                labels.append(label_map[class_folder])
    labels = np.array(labels)

    return np.array(datalist), labels, n_class
def load_test_data(test_data, test_labels, class_mapping):
    final_test_labels = [None] * sum(len(v) for v in test_labels.values())
    test_data_list = []

    for key, indices in test_labels.items():
        if key in class_mapping:
            class_label = class_mapping[key]
        else:
            raise ValueError(f"Test-set class missing from the training class mapping: {key}")

        for index in indices:
            final_test_labels[index] = class_label
            test_data_list.append(test_data[index])

    final_test_labels = np.array(final_test_labels)
    return np.array(test_data_list), final_test_labels

class OneCropsTransform:

    def __init__(self,trans_weak):
        self.trans_weak = trans_weak

    def __call__(self,x):
        x1=self.trans_weak(x)
        return [x1]

class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""

    def __init__(self, trans_weak, trans_strong0, trans_strong1):
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)

        return [x1, x2, x3]




def load_data_train(num_classes, dataset='CIFAR10', dspth='./data', bagsize=16):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
    elif dataset == 'SVHN':
        data, labels= load_svhn_data(dspth)
    elif dataset == 'MNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'MNIST', 'raw', 'train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'MNIST', 'raw', 'train-labels-idx1-ubyte')]
        n_class = num_classes
    elif dataset == 'FashionMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'FashionMNIST', 'raw', 'train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'FashionMNIST', 'raw', 'train-labels-idx1-ubyte')]
        n_class = 10
    elif dataset == 'KMNIST':
        data, labels = [], []
        datalist = [
            osp.join(dspth, 'KMNIST', 'raw', 'train-images-idx3-ubyte'),
            osp.join(dspth, 'KMNIST', 'raw', 't10k-images-idx3-ubyte')
        ]
        labelslist = [
            osp.join(dspth, 'KMNIST', 'raw', 'train-labels-idx1-ubyte'),
            osp.join(dspth, 'KMNIST', 'raw', 't10k-labels-idx1-ubyte')
        ]
        n_class = 10
    elif dataset == 'EMNISTBalanced':
        data, labels = [], []
        datalist = [osp.join(dspth, 'EMNIST','raw', 'emnist-balanced-train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST','raw', 'emnist-balanced-train-labels-idx1-ubyte')]
        n_class = 47
    elif dataset == 'AGNEWS':
        data, labels = [], []
        datalist = [
            osp.join(dspth, 'AGNEWS', 'train.csv'),
            osp.join(dspth, 'AGNEWS', 'test.csv')
        ]
        labelslist = None
        n_class = 4
    elif dataset == 'TinyImageNet':
        train_data, train_labels, n_class = load_tiny_imagenet_data(dspth)
    elif dataset == 'miniImageNet':

        train_data, train_labels = merge_train_val_test(dspth)
        subset_data_list = []
        subset_labels_list = []

        n_class = 100
        for i in range(0, len(train_data), 600):
            # Get the first 500 samples from the current chunk
            chunk_data = train_data[i:i + 600][:500]
            chunk_labels = np.array(train_labels[i:i + 600][:500])

            # Append the data and labels to the lists
            subset_data_list.append(chunk_data)
            subset_labels_list.append(chunk_labels)

        # Concatenate the subsets into final arrays
        train_data = np.concatenate(subset_data_list, axis=0)
        train_labels = np.concatenate(subset_labels_list, axis=0)
    else:
        raise ValueError("Unsupported dataset")

    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
    elif dataset in ['MNIST']:
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if n_class == 2:
            labels = np.where(np.isin(labels, [0, 2, 4, 6, 8]), 0, 1)
    elif dataset in ['FashionMNIST']:
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
    elif dataset in ['KMNIST']:

        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)


    elif dataset == 'EMNISTBalanced':
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)
                fr_label.read(8)
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
    elif dataset == 'TinyImageNet':
        data=train_data
        labels=train_labels
    elif dataset == 'miniImageNet':
        data = train_data
        labels = train_labels
    elif dataset == 'AGNEWS':
        for data_path in datalist:
            with open(data_path, 'r', encoding='utf-8') as fr:
                df = pd.read_csv(fr, header=None, names=["Class", "Title", "Description"])
                data.append((df["Title"] + " " + df["Description"]).tolist())
                labels.append((df["Class"] - 1).tolist())

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

    dataset_length=len(data)
    num_bags = len(data) // bagsize
    data_length=num_bags*bagsize
    random_indices = np.arange(data_length)
    np.random.shuffle(random_indices)

    data = data[random_indices]
    labels = labels[random_indices]
    data_u, label_prob = [], []
    labels_real = []
    labels_idx = []

    indices = np.arange(data_length)

    np.random.shuffle(indices)
    num_bags = len(indices) // bagsize
    indices_u = []
    for j in range(num_bags):
        bag_indices = indices[j * bagsize: (j + 1) * bagsize]
        if dataset in ['MNIST', 'FashionMNIST','EMNISTBalanced','KMNIST']:
            bag_data = [data[i].reshape(28, 28) for i in bag_indices]
        elif dataset == 'SVHN':
            bag_data = [data[i] for i in bag_indices]
        elif dataset == 'TinyImageNet':
            bag_data = [data[i] for i in bag_indices]
        elif dataset == 'miniImageNet':
            bag_data = [data[i] for i in bag_indices]
        else:
            bag_data = [data[i].reshape(3, 32, 32).transpose(1, 2, 0) for i in bag_indices]
        bag_labels = np.array([labels[i] for i in bag_indices])
        label_counts = Counter(bag_labels)
        labels_real.append(bag_labels)
        labels_idx.append(bag_indices)
        label_counts = OrderedDict(sorted(label_counts.items()))
        label_proportions = [label_counts.get(label, 0) / len(bag_labels) for label in range(0, num_classes)]
        data_u.append(bag_data)
        indices_u.append(j)
        label_prob.append(label_proportions)
    return data_u, label_prob, labels_real ,labels_idx,dataset_length,indices_u



def load_data_val(dataset, dspth='./data',n_classes=10):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar-100-python', 'test')
        ]
    elif dataset == 'SVHN':
        data, labels= load_svhn_val(dspth)
    elif dataset == "TinyImageNet":
        data, labels, n_class = load_tiny_imagenet_val(dspth)
    elif dataset == 'miniImageNet':
        train_data, train_labels = merge_train_val_test(dspth)
        test_data_list = []
        test_labels_list = []
        n_class = 100
        for i in range(0, len(train_data), 600):
            # Get the last 100 samples from the current chunk
            chunk_data = train_data[i:i + 600][-100:]
            chunk_labels = np.array(train_labels[i:i + 600][-100:])

            # Append the data and labels to the lists
            test_data_list.append(chunk_data)
            test_labels_list.append(chunk_labels)

        # Concatenate the subsets into final arrays
        data = np.concatenate(test_data_list, axis=0)
        labels = np.concatenate(test_labels_list, axis=0)


    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        data, labels = [], []
        for data_batch in datalist:
            with open(data_batch, 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
                data.append(entry['data'])
                labels.append(lbs)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(3, 32, 32).transpose(1, 2, 0)
            for el in data
        ]
    elif dataset == 'MNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'MNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'MNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = n_classes
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        if n_class == 2:
            labels = np.where(np.isin(labels, [0, 2, 4, 6, 8]), 0, 1)
        data = [
            el.reshape(28, 28)
            for el in data
        ]
    elif dataset == 'FashionMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'FashionMNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'FashionMNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(28, 28)
            for el in data
        ]
    elif dataset == 'KMNIST':
        data, labels = [], []
        datalist = [osp.join(dspth, 'KMNIST', 'raw', 't10k-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'KMNIST', 'raw', 't10k-labels-idx1-ubyte')]
        n_class = 10
        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # Skip the header
                fr_label.read(8)  # Skip the header
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 784))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [
            el.reshape(28, 28)
            for el in data
        ]


    elif dataset == 'EMNISTBalanced':
        data, labels = [], []
        datalist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-labels-idx1-ubyte')]
        n_class = 47

        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)
                fr_label.read(8)
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 28 * 28))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]

    return data, labels

def load_svhn_val(dspth='./data/svhn'):
    svhn_path = osp.join(dspth, 'svhn')
    with open(osp.join(svhn_path, 'test_32x32.mat'), 'rb') as fr:
        svhn_data = sio.loadmat(fr)
        data = svhn_data['X']
        labels = svhn_data['y']
    data = np.transpose(data, (3, 0, 1, 2))

    labels = labels % 10
    labels = labels.squeeze()

    return data, labels
def load_svhn_data(dspth):
    svhn_path = osp.join(dspth, 'svhn')

    with open(osp.join(svhn_path, 'train_32x32.mat'), 'rb') as fr:
        svhn_train = sio.loadmat(fr)
        train_data = svhn_train['X']
        train_labels = svhn_train['y']


    train_data = np.transpose(train_data, (3, 0, 1, 2))

    train_labels = (train_labels ) % 10

    train_labels = train_labels.squeeze()

    return train_data, train_labels



def compute_mean_var():
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class Cifar(Dataset):
    def __init__(self, dataset, data, labels, labels_real,labels_idx,indices_u, mode):
        super(Cifar, self).__init__()
        self.data, self.labels, self.labels_real,self.labels_idx,self.indices_u = data, labels, labels_real,labels_idx,indices_u
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == 'FashionMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'EMNISTBalanced':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'MNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'KMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset =='miniImageNet':
            mean, std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['FashionMNIST','KMNIST']:
            trans_weak = T1.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['MNIST','EMNISTBalanced']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale_range=(0.9, 1.1)
                ),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale_range=(0.9, 1.1)
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['TinyImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['miniImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_DLLP':
            self.trans = OneCropsTransform(trans_weak)
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST','KMNIST']:
                self.trans = T.Compose([
                    T1.Resize((28, 28)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            elif dataset in ['CIFAR10', 'CIFAR100']:
                self.trans = T.Compose([
                    T.Resize((32, 32)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        ims, lb_prob,lb_idx,indices_u = self.data[idx], self.labels[idx],self.labels_idx[idx],self.indices_u[idx]
        labels = self.labels_real[idx]
        if self.mode == 'train_u_co':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, labels,lb_idx,indices_u
        elif self.mode == 'train_u_L^2P-AHIL':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, labels,lb_idx,indices_u
        elif self.mode == 'train_u_DLLP':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx,indices_u


    def __len__(self):
        leng = len(self.data)
        return leng

class SVHN(Dataset):
    def __init__(self, dataset, data, labels, labels_real, labels_idx, indices_u, mode):
        super(SVHN, self).__init__()
        self.data, self.labels, self.labels_real, self.labels_idx, self.indices_u = data, labels, labels_real, labels_idx, indices_u
        self.mode = mode
        assert len(self.data) == len(self.labels)

        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)  # SVHN uses different mean and std

        trans_weak = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.RandomAffine(
                degrees=15,
                translate=(0.125, 0.125)),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])

        trans_strong0 = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            RandomAugment(3, 5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])

        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        elif self.mode == 'train_u_DLLP':
            self.trans = OneCropsTransform(trans_weak)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST']:
                self.trans = T.Compose([
                    T1.Resize((64, 64)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        ims, lb_prob, lb_idx, indices_u = self.data[idx], self.labels[idx], self.labels_idx[idx], self.indices_u[idx]
        labels = self.labels_real[idx]
        if self.mode == 'train_u_co':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            x_strong1 = torch.stack([self.trans(im)[2] for im in ims])
            ims_transformed = [x_weak, x_strong0, x_strong1]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == 'train_u_L^2P-AHIL':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            x_strong0 = torch.stack([self.trans(im)[1] for im in ims])
            ims_transformed = [x_weak, x_strong0]
            return ims_transformed, lb_prob, labels, lb_idx, indices_u
        elif self.mode == 'train_u_DLLP':
            x_weak = torch.stack([self.trans(im)[0] for im in ims])
            ims_transformed = [x_weak]
            return ims_transformed, lb_prob, labels, lb_idx,indices_u

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(classes,dataset, batch_size, bag_size, root='data', method='co',supervised=False):
    data_u, label_prob, labels,label_idx,dataset_length,indices_u = load_data_train(classes, dataset=dataset, dspth=root, bagsize=bag_size)
    if dataset != 'SVHN':
        ds_u = Cifar(
                dataset=dataset,
                data=data_u,
                labels=label_prob,
                labels_real=labels,
                labels_idx=label_idx,
                indices_u=indices_u,
                mode='train_u_%s' % method
            )
    else:
        ds_u = SVHN(
            dataset=dataset,
            data=data_u,
            labels=label_prob,
            labels_real=labels,
            labels_idx=label_idx,
            indices_u=indices_u,
            mode='train_u_%s' % method
        )
    #sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    sampler_u = RandomSampler(ds_u, replacement=False)
    batch_sampler_u = BatchSampler(sampler_u, batch_size, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=16,
        pin_memory=True
    )
    return dl_u,dataset_length


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='data',n_classes=10):
    data, labels = load_data_val(dataset, dspth=root,n_classes=n_classes)
    if dataset !='SVHN':
        ds = Cifar2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    else:
        ds = SVHN2(
            dataset=dataset,
            data=data,
            labels=labels,
            mode='test'
        )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class SVHN2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(SVHN2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)

        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)
        trans_weak = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),

            RandomAugment(2, 10),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST']:
                self.trans = T.Compose([
                    T1.Resize((64, 64)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


class Cifar2(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Cifar2, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif dataset == 'FashionMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'EMNISTBalanced':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'MNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset == 'KMNIST':
            mean, std = (0.1307), (0.3081)
        elif dataset =='miniImageNet':
            mean, std=(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['FashionMNIST','KMNIST']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomHorizontalFlip(p=0.5),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['MNIST', 'EMNISTBalanced']:
            trans_weak = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale_range=(0.9, 1.1)
                ),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale_range=(0.9, 1.1)
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['TinyImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset in ['miniImageNet']:
            trans_weak = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T.Resize((64, 64)),
                T.PadandRandomCrop(border=4, cropsize=(64, 64)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_co':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)
        elif self.mode == 'train_u_L^2P-AHIL':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)
        else:
            if dataset in ['MNIST', 'EMNISTBalanced', 'FashionMNIST','KMNIST']:
                self.trans = T.Compose([
                    T1.Resize((28, 28)),
                    T1.Normalize(mean, std),
                    T1.ToTensor(),
                ])
            elif dataset in ['CIFAR10', 'CIFAR100']:
                self.trans = T.Compose([
                    T.Resize((32, 32)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])
            else:
                self.trans = T.Compose([
                    T.Resize((64, 64)),
                    T.Normalize(mean, std),
                    T.ToTensor(),
                ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng
