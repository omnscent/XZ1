import torch
from torch.nn import functional as F

def Compose_loss(y_hat, y, y_tilde, w, labels):
    if labels==True:
        loss1 = -torch.log(y_hat[range(len(y_hat)), y])
    loss2 = F.layer_norm
