############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10
from loss import TotalCodingRate
from func import chunk_avg, accuracy
from lars import LARS, LARSWrapper
from func import WeightedKNNClassifier
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import datetime

######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Unsupervised Learning')

parser.add_argument('--patch_sim', type=int, default=200,
                    help='coefficient of cosine similarity (default: 200)')
parser.add_argument('--tcr', type=int, default=1,
                    help='coefficient of tcr (default: 1)')
parser.add_argument('--num_patches', type=int, default=100,
                    help='number of patches used in EMP-SSL (default: 100)')
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
parser.add_argument('--dir', type=str, default="EMP-linear",
                    help='directory name (default: EMP-linear)')     
parser.add_argument('--data', type=str, default="cifar10",
                    help='data (default: cifar10)')          
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)') 
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='GPU index (default: 0)')
parser.add_argument('--num_exps', type=int, default=1,
                    help='number of CL experiences (default: 1)')
parser.add_argument('--test_patches', type=int, default=1,
                    help='number of patches used in testing (default: 128)')
parser.add_argument('--eval_every', help='evaluate at the end of every epoch and exp', action='store_true')
parser.add_argument('--lr_linear', type=float, default=0.01,
                    help='learning rate for the linear layer (default: 0.01)')
parser.add_argument('--passes_linear', type=int, default=5,
                    help='tr passes of the linear layer for each minibatch')
args = parser.parse_args()

print(args)
num_patches = args.num_patches

str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
dir_name = f"./logs/{args.dir}/{str_now}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}_numexps{args.num_exps}_{args.msg}"

# save hyperparameters
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    with open(os.path.join(dir_name, "hyperparameters.txt"), "w") as f:
        f.write(str(args))

# Results dir
results_dir = os.path.join(dir_name, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Device
if torch.cuda.is_available():       
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    if args.gpu_idx < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu_idx}")
    else:
        device = torch.device("cuda")
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



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


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out
    
def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches,dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss/num_patches
    return loss

######################
## Prepare Training ##
######################
torch.multiprocessing.set_sharing_strategy('file_system')

if args.data == "imagenet100" or args.data == "imagenet":
    exps_trainset, _ = load_dataset("imagenet", train=True, num_patch = num_patches, num_exps=args.num_exps)

else:
    exps_trainset, _ = load_dataset(args.data, train=True, num_patch = num_patches, num_exps=args.num_exps)


# use_cuda = True
# device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = encoder(arch = args.arch)
# net = nn.DataParallel(net)
net.to(device)
if args.data == "cifar10":
    num_classes = 10
elif args.data == "imagenet100" or "cifar100":
    num_classes = 100
elif args.data == "imagenet":
    num_classes = 1000

LL = nn.Linear(4096, num_classes).to(device)
opt_linear = torch.optim.SGD(LL.parameters(), lr=args.lr_linear, momentum=0.9, weight_decay=5e-5)
criterion_linear = torch.nn.CrossEntropyLoss()



opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True,)

# scaler = GradScaler()
# if args.data == "imagenet-100":
#     num_converge = (150000//args.bs)*args.epoch
# else:
#     num_converge = (50000//args.bs)*args.epoch
    
# scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)

# Loss
contractive_loss = Similarity_Loss()
criterion = TotalCodingRate(eps=args.eps)


##############
## Training ##
##############
def main():

    for exp_idx, exp_trainset in enumerate(exps_trainset):
        dataloader = DataLoader(exp_trainset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)
        for epoch in range(args.epoch):            
            for step, step_data in tqdm(enumerate(dataloader)):
                if args.num_exps == 1:
                    data, label = step_data
                else:
                    data, label, _ = step_data


                net.zero_grad()
                opt.zero_grad()
            
                data = torch.cat(data, dim=0) 
                data = data.to(device)
                label = label.to(device)
                z_proj, z_pre, z_feat = net(data)
                
                z_list = z_proj.chunk(num_patches, dim=0)
                z_avg = chunk_avg(z_proj, num_patches)
                
                
                #Contractive Loss
                loss_contract, _ = contractive_loss(z_list, z_avg)
                loss_TCR = cal_TCR(z_proj, criterion, num_patches)
                
                loss = args.patch_sim*loss_contract + args.tcr*loss_TCR
            
                loss.backward()
                opt.step()
                # scheduler.step()

                for stp in range(args.passes_linear):
                    z_pre_avg = chunk_avg(z_pre.detach(), num_patches)
                    LL.zero_grad()
                    logits = LL(z_pre_avg)
                    loss_linear = criterion_linear(logits, label)
                    LL.zero_grad()
                    loss_linear.backward()
                    opt_linear.step()


            print("At exp:", exp_idx, ", epoch:", epoch, "loss similarity is", loss_contract.item(), ",loss TCR is:", (loss_TCR).item(), "and learning rate is:", opt.param_groups[0]['lr'])

                
            if args.eval_every:
                test_accuracy = test(net, LL, exp_idx, epoch)
                net.train()
                print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')

            model_dir = dir_name+"/save_models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save({'net': net.state_dict(),
                        'll': LL.state_dict(),
                        'opt': opt.state_dict(),
            }, f'{model_dir}exp{exp_idx}_ep{epoch}.pt')

    if not args.eval_every:
        test_accuracy = test(net, LL, exp_idx, epoch)
        net.train()
        print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')
            
        

def test(net, LL, exp_idx, epoch):
    net.eval()
    LL.eval()

    if args.data == "imagenet100" or args.data == "imagenet":
        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches, model=args.model)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

    else:
        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)
    
    
    top1_accuracy = 0
    with torch.no_grad():                
        for counter, (x, y) in tqdm(enumerate(test_loader)):

            x = torch.cat(x, dim = 0).to(device)
            y = y.to(device)

            z_proj, z_pre, z_feat = net(x)
            z_pre = chunk_avg(z_pre, args.test_patches)
            logits = LL(z_pre)

            top1, top5 = accuracy(logits, y, topk=(1,5))
            top1_accuracy += top1[0]

        top1_accuracy /= (counter + 1)           
           
        
    with open(os.path.join(results_dir, f'test_patches{args.test_patches}.txt'), 'a') as f:
                f.write(f'Exp {exp_idx}, epoch {epoch}: {top1_accuracy:.2f}\n')
    return top1_accuracy   
                


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
