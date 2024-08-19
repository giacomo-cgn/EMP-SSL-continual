############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from model.model import encoderEMP
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
import time

from src.deep_ncm import incremental_NCM_classifier

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
parser.add_argument('--dir', type=str, default="DEEP-NCM-EMP",
                    help='directory name (default: EMP-SSL-Training)')     
parser.add_argument('--data', type=str, default="cifar10",
                    help='data (default: cifar10)')          
parser.add_argument('--epoch', type=int, default=30,
                    help='max number of epochs to finish (default: 30)')
parser.add_argument('--num_exps', type=int, default=20,
                    help='number of CL experiences (default: 20)')
parser.add_argument('--ncm_momentum', type=int, default=0.999,
                    help='Update momentum value for the NCM classifier (default: 0.999) ')
parser.add_argument('--test_patches', type=int, default=1,
                    help='number of patches used in testing (default: 128)')
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='gpu index (default: 0)')
parser.add_argument('--ncm_decay', type=float, default=0.9,
                    help='decay for NCM classifier (default: 0.9)')
parser.add_argument('--ncm_lambda', type=float, default=0.5,
                    help='lambda for NCM classifier (default: 0.5)')

args = parser.parse_args()

print(args)

date = time.localtime()
date = f"{date.tm_mday}-{date.tm_mon}-{date.tm_year}_{date.tm_hour}:{date.tm_min}"

num_patches = args.num_patches
dir_name = f"./logs/{args.dir}/{date}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}__numexps{args.num_exps}_{args.msg}_lambda{args.ncm_lambda}"

# Write args to config.txt file
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    with open(f"{dir_name}/config.txt", "w") as f:
        f.write(str(args))

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
    exps_trainset, train_dataset = load_dataset("imagenet", num_exps=args.num_exps, train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)

else:
    exps_trainset, train_dataset = load_dataset(args.data, num_exps=args.num_exps, train=True, num_patch = num_patches)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)
    
    
net = encoderEMP(arch = args.arch)
net = nn.DataParallel(net)
net.to(device)

# Init deep NCM classifier
ncm_clf = incremental_NCM_classifier(features=512, alpha=args.ncm_decay)
ncm_clf.to(device)
criterion_ncm = nn.CrossEntropyLoss()


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

    net.train()
    ncm_clf.train()

    print('len exp_trainset:', len(exps_trainset))

    for exp_idx, exp_trainset in enumerate(exps_trainset):
        if args.data == "imagenet100" or args.data == "imagenet":
            dataloader = DataLoader(exp_trainset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=8)
        else:
            dataloader = DataLoader(exp_trainset, batch_size=args.bs, shuffle=True, drop_last=True,num_workers=16)

        print(f'---- Executing training loop for experience: {exp_idx} ----')
        for epoch in range(args.epoch):            
            for step, step_data in enumerate(tqdm(dataloader)):

                if args.num_exps == 1:
                    data_list, label = step_data
                else:
                    data_list, label, _ = step_data

                net.zero_grad()
                ncm_clf.zero_grad()
                opt.zero_grad()

                ncm_clf.init_from_labels(label)
            
                data = torch.cat(data_list, dim=0) 
                data = data.to(device)
                z_proj, features = net(data)
                
                z_list = z_proj.chunk(num_patches, dim=0)
                z_avg = chunk_avg(z_proj, num_patches)

                feat_list = features.chunk(num_patches, dim=0)
                feat_avg = chunk_avg(features, num_patches)
                
                
                #Contractive Loss
                loss_contract, _ = contractive_loss(z_list, z_avg)
                loss_TCR = cal_TCR(z_proj, criterion, num_patches)
                
                loss = args.patch_sim*loss_contract + args.tcr*loss_TCR



                # NCM clf
                prediction = ncm_clf(features)
                # Convert labels to match the order seen by the classifier
                label = label.repeat(args.num_patches)
                targets_converted =  ncm_clf.convert_labels(label).to(features.device)
                # Compute loss
                loss_ncm = criterion_ncm(prediction, targets_converted)

                # import matplotlib.pyplot as plt
                # plt.imshow(data[0].detach().cpu().permute(1, 2, 0))
                # plt.title(f'label {label[0].item()}')
                # plt.show()
                # plt.imshow(data[100].detach().cpu().permute(1, 2, 0))
                # plt.title(f'label {label[100].item()}')
                # plt.show()
                # plt.imshow(data[2].detach().cpu().permute(1, 2, 0))
                # plt.title(f'label {label[0].item()}')
                # plt.show()
                # plt.imshow(data[102].detach().cpu().permute(1, 2, 0))
                # plt.title(f'label {label[0].item()}')
                # plt.show()


                # loss += args.ncm_lambda*loss_ncm

                loss = loss_ncm

            
                loss.backward()
                opt.step()
                # scheduler.step()

                ncm_clf.update_means(features, label)


            # test_accuracy = test(net, ncm_clf)
            # print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')

            net.train()
            ncm_clf.train()
            

            model_dir = dir_name+"/save_models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save({'net': net.state_dict(),
                        'ncm_clf': ncm_clf.state_dict(),
                        'opt': opt.state_dict(),
            }, f'{model_dir}exp{exp_idx}_ep{epoch}.pt')
            
        
            print("At epoch:", epoch, "loss similarity is", loss_contract.item(), ",loss TCR is:", (loss_TCR).item(), "and learning rate is:", opt.param_groups[0]['lr'])

    test_accuracy = test(net, ncm_clf)
    print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')
       
                
################
## Evaluation ##
################
def test(net, ncm_clf):
    net.eval()
    ncm_clf.eval()

    if args.data == "imagenet100" or args.data == "imagenet":
        
        _, memory_dataset = load_dataset(args.data, train=True, num_patch = args.test_patches, model=args.model)
        memory_loader = DataLoader(memory_dataset, batch_size=50, shuffle=True, drop_last=True,num_workers=8)

        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches, model=args.model)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

    else:
        _, memory_dataset = load_dataset(args.data, train=True, num_patch = args.test_patches)
        memory_loader = DataLoader(memory_dataset, batch_size=50, shuffle=True, drop_last=True,num_workers=8)

        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
        for batch_idx, (inputs, targets) in pbar:
            inputs = torch.cat(inputs, dim = 0).to(device)
            _, features = net(inputs)
            outputs = ncm_clf(features)
            targets_converted = ncm_clf.convert_labels(targets).to(outputs.device)
            loss = criterion_ncm(outputs, targets_converted)
        
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_converted).sum().item()

            # Update the progress bar
            accuracy = 100. * correct / total if total > 0 else 0
            pbar.set_postfix({'loss': test_loss / (batch_idx + 1), 'accuracy': accuracy})
        

    acc = 100.*correct/total
    return acc




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
