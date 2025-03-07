import torch

def save_model(model, path='cnn_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='cnn_model.pth'):
    model.load_state_dict(torch.load(path))
