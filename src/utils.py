import os
import json
from glob import glob
import cv2
import torch
from tqdm import tqdm

# build dataset path
def build_path(dataset, mode):
    root_folder = '../data'
    if dataset == 'local':
        sec_folder = 'Mushrooms'
    elif dataset == 'open':
        sec_folder = 'open_Mushrooms'
    
    if mode == 'train':
        third_folder = 'trainset'
    elif mode == 'val':
        third_folder = 'valset'
    elif mode == 'test':
        third_folder = 'testset'
    return os.path.join(root_folder, sec_folder, third_folder)

# build samples into a list
def build_samples(path, label_type):
    samples = []
    if label_type == 'digit':
        mushroom_map = json.load(open(os.path.join(path.split('\\')[0], path.split('\\')[1], 'digit_label.json'), 'r', encoding='UTF-8'))
    elif label_type == 'distance':
        mushroom_map = json.load(open(os.path.join(path.split('\\')[0], path.split('\\')[1], 'distance_label.json'), 'r', encoding='UTF-8'))
    
    files = glob(path+'/*.jpg')

    for file in tqdm(files):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img)
        img = img.float()
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        label = mushroom_map.get(file.split('\\')[-1][:file.split('\\')[-1].rindex('_')])
        label = torch.LongTensor([label])
        samples.append((img, label))
    return samples

# save model and optimizer's state_dict
def save_checkpoint(filename, model, opt):
    state = {
        'model':model.state_dict(),
        'opt':opt.state_dict()
    }

    torch.save(state, filename)

