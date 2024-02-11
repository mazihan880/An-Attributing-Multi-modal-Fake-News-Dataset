import random
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import torch
from transformers import  logging
from typing import Dict
from Dataset import NewsDataset
from Model import Detection_Module
from torch.utils.data import DataLoader
import copy
from trainer import Trainer
from tester import Tester
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.detection_head import DetectionHead
import torch.nn as nn
import warnings
from utils.config import get_pscc_args


random.seed(2060)

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]







def pickle_reader(path):
    return pickle.load(open(path, "rb"))


def main(args):
    
    def load_network_weight(net, checkpoint_dir, name):
        weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
        netdict = net.state_dict()
        net_state_dict = torch.load(weight_path, map_location=args.device)
        pretrained_dict = {k: v for k, v in net_state_dict.items() if k in netdict}
        netdict.update(pretrained_dict)
        net.load_state_dict(netdict)
        print('{} weight-loading succeeds'.format(name))
        
    ####initialize DataLoader
    data_paths = {split: args.data_dir / f"{split}.pkl" for split in SPLITS}
    data = {split: pickle_reader(path) for split, path in data_paths.items()}

    
    datasets : Dict[str, NewsDataset] = {
        split: NewsDataset(split_data)
        for split, split_data in data.items()
    }
    
    for split, split_dataset in datasets.items():
        if split == "train" and args.mode==0:
            tr_size = len(split_dataset)
            print("tr_size:",tr_size)
            tr_set = DataLoader(
                split_dataset,  batch_size = args.batch_size,collate_fn = split_dataset.collate_fn,
                shuffle = True, drop_last = True,
                num_workers = 0, pin_memory = False)
        elif split == "eval" and args.mode == 0:
            dev_size=len(split_dataset)
            print("dev_size:",dev_size)
            dev_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=True,
                num_workers=0, pin_memory=False)
        elif args.mode == 1:
            test_size=len(split_dataset)
            print("test_size:",test_size)
            test_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last = True,
                num_workers=0, pin_memory=False)

    psccargs = get_pscc_args()

    FENet_name = 'HRNet'
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)

    ClsNet_name = 'DetectionHead'
    ClsNet = DetectionHead(psccargs)
    FENet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(FENet_name)
    ClsNet_checkpoint_dir = './checkpoint/{}_checkpoint'.format(ClsNet_name)

    load_network_weight(FENet, FENet_checkpoint_dir, FENet_name)
    load_network_weight(ClsNet, ClsNet_checkpoint_dir, ClsNet_name)
    for param in FENet.parameters():
        param.requires_grad = False 
    for param in ClsNet.parameters():
        param.requires_grad = True 
    
    classifier = Detection_Module(
                  hrnet=FENet,
                  cls_net=ClsNet,
                  model_dim=128,
                  bert_dim=768,
                  clip_dim=512
                  )
    classifier.to(args.device)

        
    ifexist=os.path.exists(args.output_dir)
    if not ifexist:
        os.makedirs(args.output_dir)
    
    
    if args.mode==0: #train/dev
        args_dict_tmp = vars(args)
        args_dict = copy.deepcopy(args_dict_tmp)
        with open(os.path.join(args.output_dir, f"param_{args.name}.txt"), mode="w") as f:
            f.write("============ parameters ============\n")
            print("============ parameters =============")
            for k, v in args_dict.items():
                f.write("{}: {}\n".format(k, v))
                print("{}: {}".format(k, v))
        trainer=Trainer(args, classifier, tr_set, tr_size, dev_set, dev_size)
        trainer.train()
    else: #test
        tester=Tester(args, classifier, test_set,test_size)
        tester.test()
    
        
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type = Path,
        help = "Directory to the dataset",
        default = "",
    )
    
    parser.add_argument(
        "--cache_dir",
        type = Path,
        help = "Directory to the preprocessed caches.",
        default = "./cache/",
    )
    
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./output/",
    )
    
    
    
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Directory to load the test model.",
        default="./ckpt/",
    )
  
   
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    
    
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:3"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--name", type=str, default="ex0")
    parser.add_argument("--mode", type=int, help="train:0, test:1", default=0)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

    
        
        