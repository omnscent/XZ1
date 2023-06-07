import torch
import copy
import random
import Net
import numpy as np
from torch import nn
import torchvision.transforms as transforms
from Data_Loader import CIFAR_dataset


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    compare = y_hat.type(y.dtype) == y
    return float(compare.type(y.dtype).sum())


def evaluate_accuracy(net, data, device):
    acc_sum = 0
    sam_sum = 0
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            y = batch.labels
            y_hat = net(batch)
            acc = accuracy(y_hat, y)
            acc_sum += float(acc)
            sam_sum += y.numel()
    return float(acc_sum / sam_sum)


def ramp_up(epoch, max_epochs, max_val, mult):
    return max_val * np.exp(mult * (1.0 - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, n_labeled, n_samples, mult=-5):
    max_val = max_val * (float(n_labeled) / n_samples)
    if epoch == 0:
        return 0.0
    elif epoch >= max_epochs:
        return max_val
    return ramp_up(epoch, max_epochs, max_val, mult)


def get_lr_lambda(total_epoch, step_per_epoch, max_epochs, last_epochs):
    def lr_lambda(step):
        step = step + 1
        if step <= max_epochs * step_per_epoch:
            return ramp_up(step, max_epochs * step_per_epoch, 1.0, mult=-5)
        elif step >= (total_epoch - last_epochs) * step_per_epoch:
            return ramp_up(
                total_epoch * step_per_epoch - step,
                last_epochs * step_per_epoch,
                1.0,
                mult=-12.5,
            )
        else:
            return 1.0

    return lr_lambda


def img_aug_MNIST(batch):
    tmp_batch = copy.deepcopy(batch)
    net = nn.Sequential(
        transforms.Pad(padding=4),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
        ),
        transforms.Normalize((0.49139968), (0.24703223)),
    )
    tmp_batch.imgs = net(tmp_batch.imgs)
    return tmp_batch


def img_aug_CIFAR(batch):
    tmp_batch = copy.deepcopy(batch)
    net = nn.Sequential(
        transforms.Pad(padding=2),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
        ),
        transforms.Normalize(
            (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
        ),
    )
    tmp_batch.imgs = net(tmp_batch.imgs)
    return tmp_batch


def train(
    net,
    data_module,
    train_batch_size,
    test_batch_size,
    labeled_num,
    optimizer,
    device,
    epochs_num,
    train_way="temporal",
    max_epochs=80,
    last_epochs=50,
    max_val=30,
    alpha=0.6,
    input_chann_num=1,
    output_chann_num=10,
    teacher_model=Net.Pi_model
):
    train_loss = []
    test_acc = []

    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

    labels_num = data_module.train_dataset.labels_num
    total_labeled = labels_num * labeled_num
    total_train_num = len(data_module.train_dataset)
    step_per_epoch = np.floor(total_train_num // train_batch_size)

    Z = torch.zeros(total_train_num, labels_num).to(device)
    z = torch.zeros(total_train_num, labels_num).to(device)

    train_dataloader = data_module.train_data_loader()
    labeled_dataloader = data_module.labeled_data_loader()
    test_dataloader = data_module.test_data_loader()

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(epochs_num, step_per_epoch, max_epochs, last_epochs),
    )
    if train_way == "mean_teacher":
        mean_teacher_model = teacher_model(input_chann_num,output_chann_num)
        mean_teacher_model.to(device)
    for epoch in range(epochs_num):
        print("epoch =", epoch, ":")
        w = weight_schedule(
            epoch,
            max_epochs,
            max_val,
            total_labeled,
            total_train_num,
        )
        loss_sum = 0
        sam_sum = 0
        for batch in train_dataloader:
            net.train()
            loss = 0
            batch = batch.to(device)
            if type(data_module) == CIFAR_dataset:
                batch = img_aug_CIFAR(batch)
            else:
                batch = img_aug_MNIST(batch)
            if train_way == "pi":
                logits_1 = net(batch)
                with torch.no_grad():
                    logits_2 = net(batch)
                if torch.sum(batch.labels, dim=0) != -train_batch_size:
                    loss = cross_entropy_loss(logits_1, batch.labels)
                loss += w * mse_loss(logits_1, logits_2.detach())
            elif train_way == "temporal":
                logits = net(batch)
                single_ind = batch.ind[:, 0]
                if torch.sum(batch.labels, dim=0) != -train_batch_size:
                    loss = cross_entropy_loss(logits, batch.labels)
                loss += w * mse_loss(logits, z[single_ind].detach())
                Z.scatter_(
                    0,
                    batch.ind,
                    alpha * Z[single_ind] + (1 - alpha) * logits,
                )
            elif train_way == "mean_teacher":
                logits_1 = net(batch)
                logits_2 = mean_teacher_model(batch)
                if torch.sum(batch.labels, dim=0) != -train_batch_size:
                    loss = cross_entropy_loss(logits_1, batch.labels)
                loss += w * mse_loss(logits_1, logits_2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += float(loss.sum())
            sam_sum += 1
            if train_way == "temporal":
                z = Z * (1.0 / (1.0 - alpha ** (epoch + 1)))
            elif train_way == "mean_teacher":
                for param, mean_param in zip(net.parameters(), mean_teacher_model.parameters()):
                    mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        train_loss.append(float(loss_sum / sam_sum))
        test_acc.append(evaluate_accuracy(net, test_dataloader, device))
        print("train_loss = ", train_loss[-1], ", test_acc = ", test_acc[-1])
    return train_loss, test_acc
