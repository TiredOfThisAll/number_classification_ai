import torch


from train import Net


def load_model():
    loaded_net = Net()

    loaded_net.load_state_dict(torch.load('mnist_model.pth2'))

    loaded_net.eval()
    return loaded_net
