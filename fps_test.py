import torch
import torch.nn as nn
import torch.utils.data as Data

from utils.data import *
from models import get_segmentation_model

from tqdm import tqdm
from argparse import ArgumentParser
import time 

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Evaluation of networks')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--dataset', type=str, default='sirstaug', help='choose datasets')
    parser.add_argument('--sirstaug-dir', type=str, default='data/sirst_aug',
                        help='dir of dataset')
    parser.add_argument('--mdfa-dir', type=str, default=r'data/MDFA',
                        help='dir of dataset')

    #
    # Evaluation parameters
    #
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--ngpu', type=int, default=0, help='GPU number')

    #
    # Network parameters
    #
    parser.add_argument('--net-name', type=str, default='transformer3',
                        help='net name: fcn')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = get_segmentation_model(args.net_name)
    net.to(device)
    net.eval()

    # define dataset
    if args.dataset == 'sirstaug':
        dataset = SirstAugDataset(mode='test', base_size=args.base_size)
    elif args.dataset == 'mdfa':
        dataset = MDFADataset(base_dir=args.mdfa_dir, mode='test', base_size=args.base_size)
    elif args.dataset == 'merged':
        dataset = MergedDataset(mdfa_base_dir=args.mdfa_dir,
                                sirstaug_base_dir=args.sirstaug_dir,
                                mode='test', base_size=args.base_size)
    elif args.dataset == 'v2':
        dataset = SirstAugDataset(base_dir='./data/V2', mode='test', base_size=args.base_size)
    else:
        raise NotImplementedError
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # warm up
    for i in range(10):
        data, labels = next(iter(data_loader))
        data = data.to(device)
        labels = labels.to(device)
        output = net(data)


    # fps test 
    with torch.no_grad():
        
        for i in tqdm(range(100)):
            data, labels = next(iter(data_loader))
            data = data.to(device)
            labels = labels.to(device)
            start_time = time.time()
            output = net(data)
            end_time = time.time()
        print(f"FPS: {1 / (end_time - start_time)}")
    
