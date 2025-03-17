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

label_map = {}  # 用于映射原始标签到连续整数的字典
class_mapping={}


def extract_labels_from_class_dict(class_dict):
    for class_idx, image_indices in enumerate(class_dict.values()):
        for image_index in image_indices:
            label_map[image_index] = class_idx
    # 为了保证顺序，我们根据图像索引排序
    sorted_label_map = dict(sorted(label_map.items()))
    # 现在我们可以得到一个与图像一一对应的标签列表
    labels = list(sorted_label_map.values())
    return labels


def load_mini_imagenet_data(dspth, split='train'):
    """
    加载 Mini-ImageNet 数据集。

    参数:
    - dspth: 数据集的路径
    - split: 'train', 'val', 'test' 对应不同的数据集部分

    返回:
    - data: 图像数据
    - labels: 标签数据
    """
    if split == 'train':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-train.pkl')
    elif split == 'val':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-val.pkl')
    elif split == 'test':
        pkl_file = osp.join(dspth, 'mini-imagenet-cache-test.pkl')
    else:
        raise ValueError("无效的 split 参数，应为 'train', 'val' 或 'test'")

    # 读取 .pkl 文件
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)

    # 从数据字典中获取图像和标签
    data = data_dict['image_data']  # 图像数据
    labels = data_dict['class_dict']  # 类别标签

    return data, labels


def merge_train_val_test(dspth):
    """
    加载并合并 Mini-ImageNet 数据集的训练集、验证集和测试集，并确保相同类别名称的标签保持一致。

    参数:
    - dspth: 数据集的路径

    返回:
    - merged_data: 合并后的图像数据
    - merged_labels: 合并后的标签数据
    """
    # 加载训练集、验证集和测试集
    train_data, train_labels = load_mini_imagenet_data(dspth, split='train')
    val_data, val_labels = load_mini_imagenet_data(dspth, split='val')
    test_data, test_labels = load_mini_imagenet_data(dspth, split='test')

    # 合并数据
    merged_data = np.concatenate([train_data, val_data, test_data], axis=0)

    # 创建类别映射，确保相同 key 的类别标签相同
    class_mapping = {}  # 用于存储每个类别的唯一标签
    class_label = 0  # 初始化标签计数器
    merged_labels = [None] * len(merged_data)  # 初始化标签列表，None表示还未分配标签
    m_labels={**test_labels, **train_labels, **val_labels}
    # 合并 train, val, test 的 labels 并设置相同 key 的标签相同
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

    # 读取标签文件
    with open(os.path.join(root, 'tiny-imagenet-200/val', 'val_annotations.txt'), 'r') as f:
        for line in f:
            parts = line.split('\t')
            image_name = parts[0]
            class_name = parts[1]
            bbox = list(map(int, parts[2:]))

            if class_name not in label_map:
                label_map[class_name] = n_class
                n_class += 1

            # 读取图像
            image_path = os.path.join(root, 'tiny-imagenet-200/val', 'images', image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)
            image = np.array(image)

            # 确保图像具有3个通道
            if len(image.shape) != 3 or image.shape[2] != 3:
                continue

            # 添加到数据列表和标签列表
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
    # 创建一个新的标签列表，用于存储测试数据的标签
    final_test_labels = [None] * sum(len(v) for v in test_labels.values())  # 初始化labels列表
    test_data_list = []  # 用于存储测试数据

    # 使用和训练集相同的 class_label 映射
    for key, indices in test_labels.items():
        # 获取训练集中相同类的标签索引
        if key in class_mapping:
            class_label = class_mapping[key]  # 获取对应的类别标签
        else:
            raise ValueError(f"测试数据集中找不到训练数据中的类: {key}")

        # 将测试数据中的索引与类别标签进行映射
        for index in indices:
            final_test_labels[index] = class_label  # 为每个索引赋值类别标签
            test_data_list.append(test_data[index])  # 加载测试数据

    # 将标签转换为 numpy 数组
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
            osp.join(dspth, 'KMNIST', 'raw', 't10k-images-idx3-ubyte')  # 包含训练和测试集
        ]
        labelslist = [
            osp.join(dspth, 'KMNIST', 'raw', 'train-labels-idx1-ubyte'),
            osp.join(dspth, 'KMNIST', 'raw', 't10k-labels-idx1-ubyte')  # 包含训练和测试集
        ]
        n_class = 10
    elif dataset == 'EMNISTBalanced':
        data, labels = [], []
        # 更新文件路径以指向EMNIST Balanced数据集的文件
        datalist = [osp.join(dspth, 'EMNIST','raw', 'emnist-balanced-train-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST','raw', 'emnist-balanced-train-labels-idx1-ubyte')]
        n_class = 47
    elif dataset == 'AGNEWS':
        data, labels = [], []
        datalist = [
            osp.join(dspth, 'AGNEWS', 'train.csv'),  # 训练集
            osp.join(dspth, 'AGNEWS', 'test.csv')  # 测试集
        ]
        labelslist = None  # AG News 数据集的标签已经嵌入文件中
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
                fr_data.read(16)  # 跳过头部信息
                fr_label.read(8)  # 跳过头部信息
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
                # 合并标题和描述
                data.append((df["Title"] + " " + df["Description"]).tolist())
                # 标签调整为从 0 开始
                labels.append((df["Class"] - 1).tolist())

        # 将列表数据拼接为单个数组
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

    dataset_length=len(data)
    num_bags = len(data) // bagsize
    data_length=num_bags*bagsize
    random_indices = np.arange(data_length)
    np.random.shuffle(random_indices)

    # 使用随机索引对 data 和 labels 进行打乱
    data = data[random_indices]
    labels = labels[random_indices]
    data_u, label_prob = [], []
    labels_real = []
    labels_idx = []

    # 使用np.arange生成初始索引序列，数量等于data的长度
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
        # 加载miniImageNet数据集的训练、验证和测试集
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
        # 使用和训练集相同的 class_label 映射


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
        # 更新为EMNIST Balanced数据集的文件路径
        datalist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-images-idx3-ubyte')]
        labelslist = [osp.join(dspth, 'EMNIST', 'raw', 'emnist-balanced-test-labels-idx1-ubyte')]
        n_class = 47  # EMNIST Balanced有47个类别

        for data_path, label_path in zip(datalist, labelslist):
            with open(data_path, 'rb') as fr_data, open(label_path, 'rb') as fr_label:
                fr_data.read(16)  # 跳过头部信息
                fr_label.read(8)  # 跳过头部信息
                data.append(np.frombuffer(fr_data.read(), dtype=np.uint8).reshape(-1, 28 * 28))
                labels.append(np.frombuffer(fr_label.read(), dtype=np.uint8))

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        data = [el.reshape(28, 28) for el in data]  # 将每个样本重塑为28x28

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

    # 加载训练数据
    with open(osp.join(svhn_path, 'train_32x32.mat'), 'rb') as fr:
        svhn_train = sio.loadmat(fr)
        train_data = svhn_train['X']
        train_labels = svhn_train['y']


    # 转换数据维度
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    # 调整标签（从1-10改为0-9）
    train_labels = (train_labels ) % 10

    # 压缩标签数组
    train_labels = train_labels.squeeze()

    # 合并训练数据和额外数据
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
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale=(0.9, 1.1)  # 0.9到1.1倍的缩放
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
        # 获取一组图片和对应的标签
        ims, lb_prob,lb_idx,indices_u = self.data[idx], self.labels[idx],self.labels_idx[idx],self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
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
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
            T.RandomAffine(
                degrees=15,  # +/- 5度的旋转
                translate=(0.125, 0.125)),
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
        ])

        # 定义强数据增强方法（trans_strong0）
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强
            RandomAugment(3, 5),
            # 自定义数据增强方法（在代码中未提供具体实现）
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
        ])

        # 定义更强的数据增强方法（trans_strong1）
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),  # 将张量转换为图像
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),  # 随机裁剪和缩放，增强数据多样性
            # 删除水平翻转操作
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 随机颜色调整，增加数据多样性
            ], p=0.8),  # 80%的概率应用颜色调整
            transforms.RandomGrayscale(p=0.2),  # 20%的概率转换为灰度图像
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
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
        # 获取一组图片和对应的标签
        ims, lb_prob, lb_idx, indices_u = self.data[idx], self.labels[idx], self.labels_idx[idx], self.indices_u[idx]
        labels = self.labels_real[idx]
        # 对图片进行变换，这里假设使用了名为 self.trans 的图像变换函数
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

        # 根据 SVHN 数据集的均值和标准差进行设置
        mean, std = (0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744)
        trans_weak = T.Compose([
            T.Resize((32, 32)),
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((32, 32)),  # 调整图像大小为 32x32 像素
            T.PadandRandomCrop(border=4, cropsize=(32, 32)),  # 添加填充并随机裁剪，用于数据增强

            RandomAugment(2, 10),
            # 自定义数据增强方法（在代码中未提供具体实现）
            T.Normalize(mean, std),  # 标准化图像（mean和std是均值和标准差）
            T.ToTensor(),  # 将图像转换为张量
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
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong0 = T.Compose([
                T1.Resize((28, 28)),
                T1.PadandRandomCrop(border=4, cropsize=(28, 28)),
                T1.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale_range=(0.9, 1.1)  # 0.9到1.1倍的缩放
                ),
                RandomAugment1(2, 10),
                T1.Normalize(mean, std),
                transforms.ToTensor(),
            ])
            trans_strong1 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
                transforms.RandomAffine(
                    degrees=15,  # +/- 5度的旋转
                    translate=(0.1, 0.1),  # 最多10%的水平和垂直平移
                    scale=(0.9, 1.1)  # 0.9到1.1倍的缩放
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
