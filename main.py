import torch
import Net
import train
import numpy as np
from torch import nn
from Data_Loader import *

"""
预设定参数
"""
input_chann_num = 1
output_chann_num = 10
net = Net.Pi_model(input_chann_num,output_chann_num)
# net = Net.ResNet18(input_chann_num,output_chann_num)
# net = Net.cifar_shakeshake26(input_chann_num,output_chann_num)
train_batch_size = 100
test_batch_size = 100
num_epochs = 300
lr = 3e-3
labeled_num = 100
data_module = MNIST_dataset(train_batch_size, test_batch_size, labeled_num, resize=32)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas = (0.9, 0.999))
# device = torch.device("mps")
device = torch.device("cuda")

"""
训练设置
"""

net.to(device)
train_loss, test_acc = train.train(
    net, data_module, train_batch_size, test_batch_size, labeled_num, optimizer, device, num_epochs, "pi", max_val=100,
)

np.savetxt('Pi_train_loss.txt',train_loss)
np.savetxt('Pi_test_acc.txt',test_acc)

# train_loss, test_acc = train.train(
#     net, data_module, train_batch_size, test_batch_size, labeled_num, optimizer, device, num_epochs, train_way="temporal",
# )

# np.savetxt('temporal_train_loss.txt',train_loss)
# np.savetxt('temporal_test_acc.txt',test_acc)

# train_loss, test_acc = train.train(
#     net, data_module, train_batch_size, test_batch_size, labeled_num, optimizer, device, num_epochs, train_way="mean_teacher", alpha=0.999, input_chann_num=input_chann_num, teacher_model=Net.Pi_model
# )

# np.savetxt('mean_teacher_train_loss.txt',train_loss)
# np.savetxt('mean_teacher_test_acc.txt',test_acc)

# train_loss, test_acc = train.train(
#     net, data_module, train_batch_size, test_batch_size, labeled_num, optimizer, device, num_epochs, train_way="mean_teacher", alpha=0.999, input_chann_num=input_chann_num, teacher_model=Net.cifar_shakeshake26
# )

# np.savetxt('mean_teacher_ResNet_train_loss.txt',train_loss)
# np.savetxt('mean_teacher_ResNet_test_acc.txt',test_acc)

# print(train_loss)
# print(test_acc)
# np.savetxt('train_loss.txt',train_loss)
# np.savetxt('test_acc.txt',test_acc)