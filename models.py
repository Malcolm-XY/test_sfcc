# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3DModel(nn.Module):
    def __init__(self, channels=1):
        super(CNN3DModel, self).__init__()

        # 第一层卷积 + 池化层：输入 (1, 9, 9, 9)，卷积输出 (32, 9, 9, 9)，池化后输出 (32, 4, 4, 4)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)  # 池化后输出 (32, 4, 4, 4)

        # 第二层卷积 + 池化层：输入 (32, 4, 4, 4)，卷积输出 (64, 4, 4, 4)，池化后输出 (64, 2, 2, 2)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)  # 池化后输出 (64, 2, 2, 2)
        
        # 第三层卷积 + 池化层：输入 (64, 2, 2, 2)，卷积输出 (128, 2, 2, 2)，池化后输出 (128, 1, 1, 1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)  # 池化后输出 (128, 1, 1, 1)

        # 全连接层 1：输入 128 * 1 * 1 * 1 = 128，输出 64 # Dropout层
        self.fc1 = nn.Linear(in_features=128 * 1 * 1 * 1, out_features=64)
        self.dropout1 = nn.Dropout(p=0.25)
        
        # 全连接层 2：输入 64，输出 32 # Dropout层
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.dropout2 = nn.Dropout(p=0.25)
        
        # 输出层：输出 3 个分类（用于 3 类分类任务）
        self.fc3 = nn.Linear(in_features=64, out_features=3)

    def forward(self, x):

        # 第一层卷积 + 激活 + 池化
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二层卷积 + 激活 + 池化
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # 第三层卷积 + 激活 + 池化
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # 展平层：将特征展平成向量，用于全连接层输入
        x = x.view(-1, 128 * 1 * 1 * 1)  # 展平为一维向量

        # 全连接层 1 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 全连接层 2 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 输出层（不使用激活函数，因为通常使用 CrossEntropyLoss）
        x = self.fc3(x)

        return x
    
class EnhancedCNN2DModel1(nn.Module):
    def __init__(self, channels=1):
        super(EnhancedCNN2DModel1, self).__init__()

        # 第一层卷积 + BatchNorm + 池化层
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 第二层卷积 + BatchNorm + 池化层
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 第三层卷积 + BatchNorm + 池化层
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 全连接层 1
        self.fc1 = nn.Linear(in_features=256 * 1 * 1 * 1, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)

        # 全连接层 2
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(p=0.25)

        # 输出层
        self.fc3 = nn.Linear(in_features=64, out_features=3)

    def forward(self, x):
        # 第一层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 展平层
        x = x.view(-1, 256 * 1 * 1 * 1)

        # 全连接层 1 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 全连接层 2 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 输出层
        x = self.fc3(x)

        return x

class EnhancedCNN3DModel1(nn.Module):
    def __init__(self, channels=1):
        super(EnhancedCNN3DModel1, self).__init__()

        # 第一层卷积 + BatchNorm + 池化层
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        # 第二层卷积 + BatchNorm + 池化层
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        # 第三层卷积 + BatchNorm + 池化层
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        # 全连接层 1
        self.fc1 = nn.Linear(in_features=256 * 1 * 1 * 1, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)

        # 全连接层 2
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(p=0.25)

        # 输出层
        self.fc3 = nn.Linear(in_features=64, out_features=3)

    def forward(self, x):
        # 第一层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三层卷积 + BatchNorm + 激活 + 池化
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 展平层
        x = x.view(-1, 256 * 1 * 1 * 1)

        # 全连接层 1 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 全连接层 2 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 输出层
        x = self.fc3(x)

        return x

class EnhancedCNN3DModel2(nn.Module):
    def __init__(self, channels=1):
        super(EnhancedCNN3DModel2, self).__init__()

        # Initial Conv Layer: (1, 9, 9, 9) -> (64, 9, 9, 9)
        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        
        # Residual Block 1
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        # Attention Mechanism
        self.attention1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # Pooling Layer
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # (128, 4, 4, 4)

        # Residual Block 2
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Attention Mechanism
        self.attention2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Adaptive Pooling
        self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))  # (256, 1, 1, 1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Dropout and final classification layer
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 3)  # Output layer for 3-class classification

    def forward(self, x):
        # Initial Convolution + BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Block 1
        residual = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x * self.attention1(x) + residual  # Attention & Residual connection
        x = self.pool1(x)

        # Residual Block 2
        residual = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.attention2(x) + residual  # Attention & Residual connection
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 256)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        
        # 两个子模型
        self.model2d = EnhancedCNN2DModel1(channels=3)
        self.model3d = EnhancedCNN3DModel1(channels=3)
        
        # 停止在子模型的 fc1 层之前
        self.model2d.fc1 = nn.Identity()
        self.model3d.fc1 = nn.Identity()
        
        # 拼接后的全连接层
        self.fc1_combined = nn.Linear(in_features=256 + 256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.fc3_out = nn.Linear(in_features=64, out_features=3)

    def forward(self, x2d, x3d):
        # 提取 2D 模型的特征
        x2d = self.model2d.conv1(x2d)
        x2d = self.model2d.pool1(F.relu(self.model2d.bn1(x2d)))
        x2d = self.model2d.conv2(x2d)
        x2d = self.model2d.pool2(F.relu(self.model2d.bn2(x2d)))
        x2d = self.model2d.conv3(x2d)
        x2d = self.model2d.pool3(F.relu(self.model2d.bn3(x2d)))
        x2d = x2d.view(-1, 256 * 1 * 1 * 1)

        # 提取 3D 模型的特征
        x3d = self.model3d.conv1(x3d)
        x3d = self.model3d.pool1(F.relu(self.model3d.bn1(x3d)))
        x3d = self.model3d.conv2(x3d)
        x3d = self.model3d.pool2(F.relu(self.model3d.bn2(x3d)))
        x3d = self.model3d.conv3(x3d)
        x3d = self.model3d.pool3(F.relu(self.model3d.bn3(x3d)))
        x3d = x3d.view(-1, 256 * 1 * 1 * 1)

        # 拼接 2D 和 3D 模型的输出
        combined = torch.cat((x2d, x3d), dim=1)

        # 通过联合全连接层
        x = F.relu(self.fc1_combined(combined))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3_out(x)

        return x
