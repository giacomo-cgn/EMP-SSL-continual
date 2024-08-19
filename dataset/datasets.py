import os
import numpy as np
import torchvision

from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet


def load_dataset(data_name, num_exps=1, train=True, num_patch = 4, path="./data/", seed=42, model='emp',
                 use_probing_tr_augs=True, use_probing_ts_augs=True, targets_semantic_order=False):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        data_name (str): name of the dataset
        transform_name (torchvision.transform): name of transform to be applied (see aug.py)
        use_baseline (bool): use baseline transform or augmentation transform
        train (bool): load training set or not
        contrastive (bool): whether to convert transform to multiview augmentation for contrastive learning.
        n_views (bool): number of views for contrastive learning
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    _name = data_name.lower()
    if _name == "imagenet":
        from .aug4img import ContrastiveLearningViewGenerator, SimSiamViewGenerator
    else:
        from .aug import ContrastiveLearningViewGenerator, SimSiamViewGenerator, EvalTransforms
      
    if model == 'emp':
        transform = ContrastiveLearningViewGenerator(num_patch = num_patch)
    elif model == 'simsiam':
        transform = SimSiamViewGenerator()

    if num_patch == 1 and not use_probing_tr_augs:
        print('Using no transforms for training of the probe')
        transform = EvalTransforms()

    if num_patch == 1 and not use_probing_ts_augs:
        print('Using no transforms for testing the probe')
        transform = EvalTransforms()

    print(f'train: {train}, num_patch: {num_patch}, use_tr_probe_augs: {use_probing_tr_augs}, use_ts_probe_augs: {use_probing_ts_augs}')

    
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train, download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "imagenet":
        if train:
            trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/ILSVRC2012/train100/",transform=transform)
            #trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/tiny-imagenet-200/train/",transform=transform)
        else:
            trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/ILSVRC2012/val100/",transform=transform)
            #trainset = torchvision.datasets.ImageFolder(root="/home/peter/Data/tiny-imagenet-200/val/",transform=transform)
        trainset.num_classes = 200  
        
    else:
        raise NameError("{} not found in trainset loader".format(_name))

    if num_exps == 1:
        exps_trainset = [trainset]
        
    elif num_exps > 1:
        if _name == 'cifar100':
            if targets_semantic_order:
                labels_order = get_semantic_class_order_cifar100()
                print('Using semantic order!')
                print(labels_order)
            else:
                labels_order = None
            benchmark = SplitCIFAR100(
                    n_experiences=num_exps,
                    seed=seed, # Fixed seed for reproducibility
                    return_task_id=False,
                    shuffle=True,
                    train_transform=transform,
                    fixed_class_order=labels_order
                )
            
        elif _name == 'cifar10':
            benchmark = SplitCIFAR10(
                    n_experiences=num_exps,
                    seed=seed, # Fixed seed for reproducibility
                    return_task_id=False,
                    shuffle=True,
                    train_transform=transform
                )
            
        exps_trainset = []
        for  exp in benchmark.train_stream:
            exps_trainset.append(exp.dataset)
        
    return exps_trainset, trainset

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]


def get_semantic_class_order_cifar100():
    fine_targets = list(range(100))
    # Get coarse targets for each fine target
    coarse_targets = sparse2coarse(fine_targets)

    # Sort the list of tuples of both targets by the coarse targets (coarse targets are together)
    coarse_and_fine_targets = list(zip(coarse_targets, fine_targets))
    sorted_coarse_and_fine_targets = sorted(coarse_and_fine_targets, key = lambda x: x[0])

    sorted_fine_targets = [x[1] for x in sorted_coarse_and_fine_targets]
    return sorted_fine_targets


def get_interleaved_semantic_class_order_cifar100():
    fine_targets = list(range(100))
    # Get coarse targets for each fine target
    coarse_targets = sparse2coarse(fine_targets)

    # Sort the list of tuples of both targets by the coarse targets (coarse targets are together)
    coarse_and_fine_targets = list(zip(coarse_targets, fine_targets))
    sorted_coarse_and_fine_targets = sorted(coarse_and_fine_targets, key = lambda x: x[0])

    sorted_fine_targets = [x[1] for x in sorted_coarse_and_fine_targets]
    sorted_fine_targets = sorted_fine_targets[::2] + sorted_fine_targets[1::2]
    return sorted_fine_targets



   
