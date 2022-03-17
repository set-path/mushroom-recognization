from model import *
from config import *
from utils import *
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from myCrossEntropyLoss import myCrossEntropyLoss


def train(model, data, label_type, opt, criterion):
    data, labels = data
    data = data.to(device)
    labels = labels.to(device)

    opt.zero_grad()
    out = model(data)

    if label_type == 'digit':
        pred = out.argmax(dim=1)
        loss = criterion(out, labels.reshape(-1))
    elif label_type == 'distance':
        pred = out.argmin(dim=1)
        loss = criterion(out, labels)

    if label_type == 'digit':
        acc = torch.sum(labels.cpu().view(-1) ==
                        pred.cpu().view(-1)).item()/len(labels)
    elif label_type == 'distance':
        acc = torch.sum(torch.argmin(labels.cpu(), axis=1).view(-1)
                        == pred.cpu().view(-1)).item()/len(labels)

    loss.backward()
    opt.step()

    return loss.cpu().detach().numpy(), acc


def val(model, label_type, valloader, criterion):
    avg_loss = 0
    avg_acc = 0
    idx = 0
    for batch_id, data in enumerate(valloader):
        data, labels = data
        data = data.to(device)
        labels = labels.to(device)

        out = model(data)

        if label_type == 'digit':
            pred = out.argmax(dim=1)
            loss = criterion(out, labels.reshape(-1))
        elif label_type == 'distance':
            pred = out.argmin(dim=1)
            loss = criterion(out, labels)

        avg_loss += loss.cpu().detach().numpy()

        if label_type == 'digit':
            acc = torch.sum(labels.cpu().view(-1) ==
                            pred.cpu().view(-1)).item()/len(labels)
        elif label_type == 'distance':
            acc = torch.sum(torch.argmin(labels.cpu(), axis=1).view(-1)
                            == pred.cpu().view(-1)).item()/len(labels)
        avg_acc += acc

        idx = batch_id

    return avg_loss/idx, avg_acc/idx


def main():
    # build model and optimizer
    if not load_state:
        model = build_model(model_name, pre_trained, num_classes, attention)

        opt = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=lr)
    else:
        model, opt = load_model()

    # build dataset
    trainset = Dataset(dataset, 'train', label_type)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = Dataset(dataset, 'val', label_type)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # choose loss function
    if label_type == 'digit':
        criterion = nn.CrossEntropyLoss()
    elif label_type == 'distance':
        criterion = myCrossEntropyLoss()

    max_loss = 10000

    for epoch in range(epochs):
        model.train()
        for batch_id, data in enumerate(trainloader):
            loss, acc = train(model, data, label_type, opt, criterion)
            print(
                f'[train] epoch {epoch} batch_id {batch_id} loss {loss} accuracy {acc}')

        model.eval()

        loss, acc = val(model, label_type, valloader, criterion)
        print(f'[validation] loss {loss} accuracy {acc}')

        if loss < max_loss:
            max_loss = loss
            filename = '../models/BEST_checkpoint.tar'
            save_checkpoint(filename, model, opt)

        filename = f'../models/checkpoint_{epoch}_{loss}.tar'
        save_checkpoint(filename, model, opt)


if __name__ == '__main__':
    main()
