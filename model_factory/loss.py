import torch.nn.functional as F


def nll_loss(output, target):
    print('output:', output)
    print('target:', target.shape)

    return F.nll_loss(output, target)

def CE_loss(output, target):
    target = target.long()
    return F.cross_entropy(output, target)