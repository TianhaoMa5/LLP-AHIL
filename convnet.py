
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            # 第一部分: 3x3卷积 + BN + LeakyReLU (96通道)
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # 输入通道:3(RGB图像)
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),

            # 最大池化 + BN
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(96),

            # 第二部分: 3x3卷积 + BN + LeakyReLU (192通道)
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            # 最大池化 + BN
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(192),

            # 第三部分: 3x3卷积 + BN + LeakyReLU (192通道)
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            # 1x1卷积 + BN + LeakyReLU
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),

            # 1x1卷积 + BN + LeakyReLU (10通道)
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU()
        )
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出尺寸为1x1

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # 展平
        return x


class CustomCNN(nn.Module):
    def __init__(self, n_classes):
        super(CustomCNN, self).__init__()
        # Define the layers of the CNN
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.LeakyReLU()
        )

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layer
        self.fc = nn.Linear(128, n_classes)  # Assuming the output is a 10-class classification
        self.init_weight()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

# Initialize the network
 # This will print the architecture of the network we just created.
