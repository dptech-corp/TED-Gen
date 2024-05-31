import torch.nn as nn
import torch
def re_loss(logits, targets, rl = False):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    if 3.2 in torch.abs(targets):
        x,y = torch.where(torch.abs(targets) == 3.2)
        for i in range(x.shape[0]):
            if logits[x[i],y[i]] * targets[x[i],y[i]] < 0:
                logits[x[i],y[i]] = logits[x[i],y[i]]*-1

    shift_loss = mse_loss(logits[:,:2], targets[:,:2]) + 0.5 * l1_loss(logits[:,:2], targets[:,:2])
    if rl:
        rotation_loss = mse_loss(logits[:,2], targets[:,2]) + 0.5 * l1_loss(logits[:,2], targets[:,2])
        total_loss = shift_loss + rotation_loss
    else:
        total_loss = shift_loss

    return  total_loss, shift_loss.item()

def mo_loss(logits, targets):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    rotation_loss = mse_loss(logits, targets) + 0.5 * l1_loss(logits, targets)
    total_loss = rotation_loss

    return  total_loss, rotation_loss.item()


def re3_loss(logits, targets):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    shift_loss = mse_loss(logits[:,:4], targets[:,:4]) + 0.5 * l1_loss(logits[:,:4], targets[:,:4])
    total_loss = shift_loss

    return  total_loss, shift_loss.item()