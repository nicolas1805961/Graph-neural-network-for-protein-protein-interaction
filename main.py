import torchvision
from datasets import load_dataset
from torchvision import transforms
import torch
from dataloader import MyOwnDataset
from model import GCN
from torch_geometric.loader import DataLoader
from train import train
import os
from datetime import datetime
from copy import copy
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import numpy as np
import random
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML
import logging

def read_config_video(filename):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="yaml config file", required=True)
    parser.add_argument("--deterministic",
                            help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                                "this is not necessary. Deterministic training will make you overfit to some random seed. "
                                "Don't use that.",
                            required=False, default=False, action="store_true")
    args = parser.parse_args()
    deterministic = args.deterministic

    if deterministic:
        random.seed(12345)
        np.random.seed(12345)
        torch.cuda.manual_seed_all(12345)
        torch.manual_seed(12345)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    output_folder = 'out'

    train_dataset = MyOwnDataset(root='data', file_name='train_data.csv')
    val_dataset = MyOwnDataset(root='data', file_name='val_data.csv')

    config = read_config_video(os.path.join(Path.cwd(), args.config))

    # You can lower your batch size if you're running out of GPU memory
    batch_size = config['batch_size']
    batch_size_val = config['batch_size_val']
    epochs = config['epochs']
    validation_step = config['validation_step']
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, follow_batch=['x_1', 'x_2'], shuffle=True)

    #model = get_model()
    model = GCN(input_dim=20, hidden_dim=hidden_dim, num_layers=num_layers).to('cuda:0')

    timestr = datetime.now().strftime("%Y-%m-%d_%HH%M_%Ss_%f")
    log_dir = os.path.join(copy(output_folder), timestr)
    writer = SummaryWriter(log_dir=log_dir)

    logging.basicConfig(filename='main.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)

    yaml = YAML()
    with open(os.path.join(log_dir, 'config.yaml'), 'wb') as f:
        yaml.dump(config, f)

    train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, writer=writer, epochs=epochs, validation_step=validation_step, log_dir=log_dir, logger=logger)