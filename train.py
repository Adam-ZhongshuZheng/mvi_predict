'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
from data_loader import MVIClassifyLoader
from model import USNETres, ClassifyNet, MVIBigNet, GroupFeatureNet, get_USNE

import pdb

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--max_epoch', default=100, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mode', '-m', default='A', type=str, help='train/test mode')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

flod = args.flod
mode = args.mode



# Data
def data_prepare(loadmode):
    """
    make the dataloader and return them: train, test, validation

    Args:
        loadmode (str): the mode in ['A', 'D', 'P', 'ALL']
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    readpath = os.path.join('data_flods', 'flod' + str(flod))

    train_set = MVIClassifyLoader(os.path.join(readpath, 'train_set.csv'), transform=transform_train, mode=loadmode)
    valid_set = MVIClassifyLoader(os.path.join(readpath, 'valid_set.csv'), transform=transform_test, mode=loadmode)
    test_set = MVIClassifyLoader(os.path.join(readpath, 'test_set.csv'), transform=transform_test, mode=loadmode)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # # To get shuffled dataset
    # dataset = MVIClassifyLoader(os.path.join(readpath, 'filelist.csv'), transform=transform_train, mode=loadmode)
    # dataset_test = MVIClassifyLoader(os.path.join(readpath, 'filelist.csv'), transform=transform_test, mode=loadmode)

    # valid_rate = [0.6, 0.8, 1.0]    # the valid and test part

    # indices = list(range(len(dataset)))
    # split1 = int(np.floor(valid_rate[0] * len(dataset)))
    # split2 = int(np.floor(valid_rate[1] * len(dataset)))
    # np.random.shuffle(indices)
    # train_idx, valid_idx, test_idx = indices[:split1], indices[split1: split2], indices[split2:]
    # train_sampler = SubsetRandomSampler(train_idx)
    # vali_sampler = SubsetRandomSampler(valid_idx)
    # test_sampler = SubsetRandomSampler(test_idx)

    # trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)
    # validloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=vali_sampler, shuffle=False)
    # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)

    return trainloader, validloader, testloader


# Model
def model_prepare(mode):
    """
    Args:
        mode: can be 'A/D/P' or 'ALL'
    """
    print('==> Building model..')
    global best_acc
    global start_epoch

    if mode in ['A', 'D', 'P']:
        net = get_USNE()
    elif mode in ['G']:
        net = GroupFeatureNet()
    else:
        net = MVIBigNet()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # TO Check the check point.
    if args.resume:
        print('==> Resuming model from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'mode' + mode + '.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    elif mode is 'ALL':
        print('==> Initialize the big model from checkpoints..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_a = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'modeA.t7')
        checkpoint_d = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'modeD.t7')
        checkpoint_p = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'modeP.t7')
        checkpoint_g = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'modeG.t7')
        net.feature_a.load_state_dict(checkpoint_a['net'])
        net.feature_d.load_state_dict(checkpoint_d['net'])
        net.feature_p.load_state_dict(checkpoint_p['net'])
        net.feature_g.load_state_dict(checkpoint_g['net'])

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()

    # print('==> parameters to train:')
    # for name, param in net.named_parameters():
    #     print("\t", name)

    return net, optimizer, criterion


def train(epoch, dataloader, net, optimizer, criterion, mode):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, databook in enumerate(dataloader):

        optimizer.zero_grad()
#         pdb.set_trace()

        if mode in ['A', 'D', 'P']:
            sid = databook['id']
            inputs = databook['img']
            targets = databook['mvi']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.uint8)
            outputs = net(inputs)  # one_hot_targets = make_one_hot(targets, 2)
        elif mode in ['G']:
            inputs = databook['groupfeature']
            targets = databook['mvi']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.uint8)            
        else:
            inputs_a = databook['apimg']
            inputs_d = databook['dpimg']
            inputs_p = databook['pvpimg']
            inputs_g = databook['groupfeature']
            targets = databook['mvi']
            inputs_a = inputs_a.to(device, dtype=torch.float)
            inputs_d = inputs_d.to(device, dtype=torch.float)
            inputs_p = inputs_p.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.uint8)
            outputs = net(inputs_a, inputs_d, inputs_p, inputs_g)  # one_hot_targets = make_one_hot(targets, 2)
        
        
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        # pdb.set_trace()
        # _, targets = one_hot_targets.max(1)
        total += targets.size(0)
        correct += predicted.byte().eq(targets).sum().item()

        # temp = predicted + targets
        # intersection = (temp == 2).sum().item()
        # union = (temp > 0).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # print('Loss: %.3f | Acc: %.3f (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch, dataloader, net, optimizer, criterion, mode, vali=True):
    """Validation and the test."""
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    wrong_list = []     # To get the wrong id
    with torch.no_grad():
        for batch_idx, databook in enumerate(dataloader):

            if mode in ['A', 'D', 'P']:
                sid = databook['id']
                inputs = databook['img']
                targets = databook['mvi']
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.uint8)
                outputs = net(inputs)  # one_hot_targets = make_one_hot(targets, 2)
            elif mode in ['G']:
                inputs = databook['groupfeature']
                targets = databook['mvi']
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.uint8)
            else:
                inputs_a = databook['apimg']
                inputs_d = databook['dpimg']
                inputs_p = databook['pvpimg']
                inputs_g = databook['groupfeature']
                targets = databook['mvi']
                inputs_a = inputs_a.to(device, dtype=torch.float)
                inputs_d = inputs_d.to(device, dtype=torch.float)
                inputs_p = inputs_p.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.uint8)
                outputs = net(inputs_a, inputs_d, inputs_p, inputs_g)  # one_hot_targets = make_one_hot(targets, 2)

            # one_hot_targets = one_hot_for_class(targets, 2)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            # _, targets = one_hot_targets.max(1)
            total += targets.size(0)
            correct += predicted.byte().eq(targets).sum().item()

            # To get the wrong sample id list
            for i in 1 - predicted.byte().eq(targets):
                if int(i) is 0 and sid[int(i)] not in wrong_list:
                    wrong_list.append(sid[int(i)])

            # temp = predicted + targets
            # intersection = (temp == 2).sum().item()
            # union = (temp > 0).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save the best model till now.
    if vali is True:
        # Save checkpoint.
        acc = 100. * correct / total
        if acc >= best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_flod' + str(flod) + 'mode' + mode + '.t7')
            best_acc = acc

            # Saving the wrong list
            wrong_file = open('wrong_logs/wrong_list' + str(flod) + '.adm', 'a')
            wrong_file.write('epoch:' + str(epoch) + ',acc:' + str(acc) + ',')
            for i in range(len(wrong_list)):
                wrong_file.write(str(wrong_list[i]))
                if i != len(wrong_list):
                    wrong_file.write(',')
                else:
                    wrong_file.write('\n')

            wrong_file.close()


if __name__ == '__main__':

    trainloader, valiloader, testloader = data_prepare(mode)
    net, optimizer, criterion = model_prepare(mode)
    for epoch in range(start_epoch, start_epoch+args.max_epoch):

        train(epoch, trainloader, net, optimizer, criterion, mode)
        test(epoch, valiloader, net, optimizer, criterion, mode, vali=True)
#     print(start_epoch)
#     test(start_epoch, testloader, net, optimizer, criterion, mode, vali=False)
