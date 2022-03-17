from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from myCrossEntropyLoss import myCrossEntropyLoss
from config import *
from model import *

def test():
    testset = Dataset(dataset, 'test', label_type)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model, _ = load_model()

    if label_type == 'digit':
        criterion = nn.CrossEntropyLoss()
    elif label_type == 'distance':
        criterion = myCrossEntropyLoss()

    avg_loss = 0
    avg_acc = 0
    idx = 0
    for batch_id, data in enumerate(testloader):
        data, labels = data
        data.to(device)
        labels.to(device)

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

    print(f'[test] loss {loss/idx} accuracy {acc/idx}')


if __name__=='__main__':
    test()