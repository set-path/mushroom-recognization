import timm
from SELayer import SELayer
from ECANet import ECANet
import torch
import torch.optim as optim
from config import *
import os

# build a new model
def build_model(model_name, pre_trained, num_classes, attention):
    model = timm.create_model(model_name, pretrained=pre_trained, num_classes=num_classes)

    # add attention layer
    if not attention:
        return model
    else:
        model.conv_stem.add_module('first_selayer',SELayer(in_channels=16, reduction_coefficient=2))
        model.conv_stem.add_module('sec_selayer',SELayer(in_channels=16, reduction_coefficient=2))
        model.conv_head.add_module('ecanet', ECANet(in_channels=16, kernel_size=3))
        model = freeze_layer(model)
    model = model.to(device)
    
    return model

# load model from state_dict file
def load_model():
    filename = '../models/BEST_checkpoint.tar'
    if os.path.exists(filename):
        state = torch.load(filename)

        model = build_model(model_name, pre_trained, num_classes, attention)
        model.load_state_dict(state['model'])
        model = model.to(device)

        opt = optim.Adam(filter(lambda p: p.requires_grad,
                        model.parameters()), lr=lr)
        opt.load_state_dict(state['opt'])

        return model, opt
    else:
        print('no state_dict file, please set `load_state=False` in `config.py`')
        exit(0)

# freeze other layer except attention layer and classifier layer
def freeze_layer(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv_stem.first_selayer.parameters():
        param.requires_grad = True
    for param in model.conv_stem.sec_selayer.parameters():
        param.requires_grad = True
    for param in model.conv_head.ecanet.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model