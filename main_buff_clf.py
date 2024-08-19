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

from ncm import NCMClassifier
from src.buffer import ReservoirBuffer

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
parser.add_argument('--dir', type=str, default="BUFF-CLF-EMP",
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
parser.add_argument('--buffer_size', type=int, default=2000,
                    help='size of the buffer (default: 2000)')
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='gpu index (default: 0)')
parser.add_argument('--pass_before_inference',
                    help='Recalulate features of buffer samples with updated encoder before inference (default: True)',
                    action='store_true')

args = parser.parse_args()

print(args)

date = time.localtime()
date = f"{date.tm_mday}-{date.tm_mon}-{date.tm_year}_{date.tm_hour}:{date.tm_min}"

num_patches = args.num_patches
dir_name = f"./logs/{args.dir}/{date}_numpatch{args.num_patches}_bs{args.bs}_lr{args.lr}__numexps{args.num_exps}_{args.msg}"

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


opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,nesterov=True)
opt = LARSWrapper(opt,eta=0.005,clip=True,exclude_bias_n_norm=True,)

scaler = GradScaler()
if args.data == "imagenet-100":
    num_converge = (150000//args.bs)*args.epoch
else:
    num_converge = (50000//args.bs)*args.epoch
    
# scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=num_converge, eta_min=0,last_epoch=-1)

# Loss
contractive_loss = Similarity_Loss()
criterion = TotalCodingRate(eps=args.eps)


##############
## Training ##
##############
def main():

    # Init buffer
    buffer = ReservoirBuffer(args.buffer_size, device=device, alpha_ema=0.)

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
                opt.zero_grad()
            
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
            
                loss.backward()
                opt.step()
                # scheduler.step()

                # Update buffer
                buffer.add(data_list[0].detach(), feat_list[0].detach(), label.detach()) # TODO: Prova a cambiare con feat_avg
            

            # test_accuracy = test(net, buffer)
            # print(test_accuracy)
            # print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')

            net.train()
            

            model_dir = dir_name+"/save_models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save({'net': net.state_dict(),
                        # 'ncm_clf': ncm_clf.state_dict(),
                        'opt': opt.state_dict(),
            }, f'{model_dir}exp{exp_idx}_ep{epoch}.pt')
            
        
            print("At epoch:", epoch, "loss similarity is", loss_contract.item(), ",loss TCR is:", (loss_TCR).item(), "and learning rate is:", opt.param_groups[0]['lr'])
        

    test_accuracy = test(net, buffer)
    print(f'---- Test accuracy at epoch {epoch} of experience {exp_idx} is: {test_accuracy:.2f}')
       
                
################
## Evaluation ##
################
def test(net, buffer):
    net.eval()

    # # Final pass over buffer data
    # if args.pass_before_inference:
    #     with torch.no_grad():
    #         # pass all sample through the net and update the buffer with new features
    #         step = 50
    #         step_indices = range(0, len(buffer.buffer), step)
    #         print('---- Executing forward pass on buffer ----')
    #         for start_idx in tqdm(step_indices):
    #             end_idx = min(start_idx + step, len(buffer.buffer))
    #             indices = list(range(start_idx, end_idx))
    #             data_batch, _, _ = buffer.sample_indices(indices)
    #             data_batch = data_batch.to(device)
    #             _, features = net(data_batch)
    #             buffer.update_features(features.detach(), indices)

    class_means_dict = {}
    classes_count_dict = {}
    num_els = 0    
    for sample, label, feature in zip(buffer.buffer, buffer.buffer_labels, buffer.buffer_features):

        print(feature.shape)
        num_els += sample.size(0)
        # feature = torch.nn.functional.normalize(feature, p=2, dim=0)

        if int(label.item()) not in class_means_dict.keys():
            class_means_dict[int(label.item())] = feature.cpu().detach().clone()
        else:
            class_means_dict[int(label.item())] += feature.cpu().detach().clone()

        classes_count_dict[int(label.item())] = classes_count_dict.get(label.item(), 0) + 1

    class_means_dict[int(label.item())] /= float(num_els)
    class_means_dict[int(label.item())] /= class_means_dict[int(label.item())].norm()

    for key in class_means_dict.keys():
        class_means_dict[key] /= classes_count_dict[key]

    ncm_clf = NCMClassifier(normalize=False)
    ncm_clf.class_means_dict = class_means_dict

    ncm_clf.vectorize_means_dict()
    print(f'Num prototypes in NCM: {len(ncm_clf.class_means_dict.keys())}')

    for key, val in ncm_clf.class_means_dict.items():
        print(f'Prototype {key}: {val.shape}')

    print('clf_means shape:', ncm_clf.class_means.shape)
    print('clf_means:', ncm_clf.class_means)


    if args.data == "imagenet100" or args.data == "imagenet":

        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches, model=args.model)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

    else:

        _, test_data = load_dataset(args.data, train=False, num_patch = args.test_patches)
        test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)
    
    num_correct, num_total = 0, 0
    
    with torch.no_grad():
        print('---- Executing forward pass on evaluation set ----')            
        for x, y in tqdm(test_loader):

            x = torch.cat(x, dim = 0).to(device)

            z_proj, z_pre = net(x)

            z_pre = chunk_avg(z_pre, args.test_patches)
            z_pre = z_pre.detach().cpu()

            print('z_pre:', z_pre)
           
            # NCM classifier
            distances_logits = ncm_clf(z_pre)
            print('distances logits:', distances_logits)
            _, predicted = torch.max(distances_logits, 1)

            print(f'predicted: {predicted}, label: {y}')

            print('closest_centroid: ', ncm_clf.class_means_dict[predicted[0].item()])
            print('closest distance: ', distances_logits[0, predicted[0].item()])

            correct_pred = (predicted == y).sum().item()
            num_correct += correct_pred
            num_total += x.size(0)

    test_accuracy = 100. * num_correct / num_total
    return test_accuracy




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
