import torch

# choose device to run model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train configuration
lr = 1e-4
epochs = 30
batch_size = 12

# model configuration
load_state = False # load train state info
dataset = 'local' # choose dataset: local or open
label_type = 'digit' # choose label type: digit or genetic distance
model_name = 'mobilenetv3_large_100' # choose the backbone model
pre_trained = True # load pre_trained parameters
num_classes = 18 # classification number
attention = True # add attention module
