import torch
import Net
import Optim
import numpy as np
import train
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from Data_Loader import load_mnist_data, load_Fashion_mnist_data, load_CIFAR_data

"""
预设定参数
"""

batch_size = 256
num_epochs = 50
lr = 0.01
train_iter, test_iter = load_Fashion_mnist_data(batch_size, resize=224)
# train_iter, test_iter = load_Fashion_mnist_data(batch_size, resize=32)
loss = nn.CrossEntropyLoss(reduction="none")
# device = torch.device("mps")
device = torch.device("cuda")


"""
模型设置
"""
# net = Net.Linear_Model(784, 10)
# net = Net.Multi_Linear_Model(1, 784, [256], 10)
# net = Net.Multilayer_Perceptron(3, [784, 512, 256, 64, 10])
# net = Net.LeNet()
# net = Net.AlexNet()
# net = Net.VGG([[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]])
net = Net.ResNet18(1, 10)
# net = Net.HTNet(3, 10)


"""
训练设置
"""

net.to(device)
trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# train_acu, train_loss, test_acu = net.train(
#     train_iter, test_iter, num_epochs, lr, "default"
# )
train_acu, train_loss, test_acu = train.train(
    net, train_iter, test_iter, num_epochs, loss, trainer, device, 6
)
print(train_acu)
print(train_loss)
print(test_acu)
