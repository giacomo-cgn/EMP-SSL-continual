############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoderSimSiam
from dataset.datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10
from loss import TotalCodingRate
from func import chunk_avg
from lars import LARS, LARSWrapper
from func import WeightedKNNClassifier
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast

######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')
parser.add_argument('--bs', type=int, default=100,
                    help='batch size (default: 100)')
parser.add_argument('--lr', type=float, default=0.3,
                    help='learning rate (default: 0.3)')        
parser.add_argument('--eps', type=float, default=0.2,
                    help='eps for TCR (default: 0.2)') 
parser.add_argument('--msg', type=str, default="NONE",
                    help='additional message for description (default: NONE)')     
parser.add_argument('--dir', type=str, default="SimSiam-Training",
                    help='directory name (default: EMP-SSL-Training)')     
parser.add_argument('--data', type=str, default="cifar10",
                    help='data (default: cifar10)')          
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')
parser.add_argument('--num_exps', type=int, default=20,
                    help='number of CL experiences (default: 20)')

args = parser.parse_args()

print(args)

num_patches = 2
dir_name = f"./logs/{args.dir}/bs{args.bs}_lr{args.lr}_{args.msg}"


######################
## Prepare Training ##
######################
torch.multiprocessing.set_sharing_strategy('file_system')

if args.data == "imagenet100" or args.data == "imagenet":
    exps_trainset, train_dataset = load_dataset("imagenet", num_exps=args.num_exps, train=True, num_patch = num_patches, model='simsiam')
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)

else:
    exps_trainset, train_dataset = load_dataset(args.data, num_exps=args.num_exps, train=True, num_patch = num_patches, model='simsiam')
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = encoderSimSiam(arch = args.arch)
net = nn.DataParallel(net)
net.cuda()


opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True,)

scaler = GradScaler()
if args.data == "imagenet-100":
    num_converge = (150000//args.bs)*args.epoch
else:
    num_converge = (50000//args.bs)*args.epoch
    
# scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)

# Loss
criterion = nn.CosineSimilarity(dim=1).cuda()


#####################
## Helper Function ##
#####################

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)



##############
## Training ##
##############
def main():
    assert num_patches==2, 'SimSiam can only have 2 views from each samples'

    for exp_idx, exp_trainset in enumerate(exps_trainset):
        if args.data == "imagenet100" or args.data == "imagenet":
            dataloader = DataLoader(exp_trainset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)
        else:
            dataloader = DataLoader(exp_trainset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)

        for epoch in range(args.epoch):            
            for step, step_data in enumerate(tqdm(dataloader)):

                if args.num_exps == 1:
                    data, label = step_data
                else:
                    data, label, _ = step_data
                    

                net.zero_grad()
                opt.zero_grad()
            
                data = torch.cat(data, dim=0) 
                data = data.cuda()
                feature, z_proj, p_pred = net(data)
                
                z_list = z_proj.chunk(num_patches, dim=0)
                p_list = p_pred.chunk(num_patches, dim=0)

                z1, z2 = z_list[0], z_list[1]
                p1, p2 = p_list[0], p_list[1]         
                
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5                
            
                loss.backward()
                opt.step()
                # scheduler.step()
                

            model_dir = dir_name+"/save_models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(net.state_dict(), f'{model_dir}exp{exp_idx}_ep{epoch}.pt')
            
        
            print("At epoch:", epoch, "loss is", loss.item(), "and learning rate is:", opt.param_groups[0]['lr'])
       
                


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
