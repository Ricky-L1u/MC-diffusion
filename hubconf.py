from ViT import ViT
from torch.hub import load_state_dict_from_url
dependencies = ['torch', 'math']


def general_model():
    checkpoint = 'https://github.com/Ricky-L1u/MC-diffusion/releases/download/v1/general_model.pt'
    model = ViT(1024, 1000)
    model.load_state_dict(load_state_dict_from_url(checkpoint, progress=True))
    return model


def better_model():
    checkpoint = 'https://github.com/Ricky-L1u/MC-diffusion/releases/download/v1/better_model.pt'
    model = ViT(1024, 1000)
    model.load_state_dict(load_state_dict_from_url(checkpoint, progress=True))
    return model