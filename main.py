import os
import yaml
import shutil
import argparse
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from utils import *
from models.resnet import resnet18
from models.wideresnet import WRN34_10

def training(epoch, model, dataloader, optimizer, alpha, epsilon, num_iters, gamma, beta,
             tb, num_classes, pm, lower, upper, burn_in_period, training_type, kl_lambda):
    total = 0
    total_loss = 0
    total_correct = 0
    
    model.train()
    sigmoid = nn.Sigmoid()
    xent = nn.CrossEntropyLoss()
    for batch_ind, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = pgd_attk(model, xent, inputs, targets, alpha, epsilon, num_iters, training_type, lower, upper)
        
        if pm == 'nat':
            logits = model(inputs)
            margin = calc_probabilistic_margin(logits, targets, num_classes)
        elif pm == 'adv':
            logits = model(x)
            margin = calc_probabilistic_margin(logits, targets, num_classes)
        elif pm == 'dif':
            gt_mask = torch.eye(num_classes)[targets].cuda()
            probs_nat = torch.softmax(model(inputs), dim=1)
            probs_adv = torch.softmax(model(x), dim=1)
            margin = torch.sum(probs_nat * gt_mask, dim=1) - torch.sum(probs_adv * gt_mask, dim=1)
            
        weight = sigmoid(-gamma * (margin - beta))
        weight = weight * len(weight) / torch.sum(weight)
        
        logits_adv = model(x)
        logits_nat = model(inputs)
        if epoch > burn_in_period:
            loss = calc_loss(xent, targets, logits_adv, logits_nat, weight, kl_lambda, training_type)
        else:
            loss = calc_loss(xent, targets, logits_adv, logits_nat, torch.ones_like(targets), kl_lambda, training_type)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct = torch.argmax(logits, dim=1).eq(targets).sum().item()
        total_correct += correct
        total_loss += loss.item()
        total += inputs.size(0)
        
        if batch_ind % 100 == 0:
            tb.add_scalar('train/acc', total_correct/total, (len(dataloader)*epoch)+batch_ind)
            tb.add_scalar('train/loss', total_loss/total, (len(dataloader)*epoch)+batch_ind)
            print('%d epoch [%d/%d] | loss: %.4f (avg: %.4f) | acc: %.4f (avg: %.4f)' % (epoch, batch_ind, len(dataloader), loss.item(), total_loss/total, correct/inputs.size(0), total_correct/total))
            
def evaluation(epoch, model, dataloader, alpha, epsilon, num_iters, lower, upper):
    model.eval()
    total_correct_nat = 0
    total_correct_adv = 0
    xent = nn.CrossEntropyLoss()
    for samples in dataloader:
        inputs, targets = samples[0].cuda(), samples[1].cuda()
        x = Variable(inputs, requires_grad=True)
        for _ in range(num_iters):
            x.requires_grad_()
            logits = model(x)
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
        
            x = x.data.detach() + alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, min=lower, max=upper)
        
        with torch.no_grad():
            logits_nat = model(inputs)
            logits_adv = model(x)
        
        total_correct_nat += torch.argmax(logits_nat.data, dim=1).eq(targets.data).cpu().sum()
        total_correct_adv += torch.argmax(logits_adv.data, dim=1).eq(targets.data).cpu().sum()
        
    print('Validation | nat acc: %.4f | adv acc: %.4f ' % (total_correct_nat / len(dataloader.dataset), total_correct_adv / len(dataloader.dataset)))
    return (total_correct_nat / len(dataloader.dataset)).item(), (total_correct_adv / len(dataloader.dataset)).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--config_file', type=str, default='')
    args = parser.parse_args()
    
    print(args)
    
    #####################################################
    # Loading training configures
    #####################################################
    with open(args.config_file) as yf:
        configs = yaml.safe_load(yf.read())['training']
    ## training setting
    model_type = configs['model_type']
    training_type = configs['training_type']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    num_classes = configs['num_classes']
    lr = configs['learning_rate']
    momentum = configs['momentum']
    weight_decay = configs['weight_decay']
    burn_in_period = configs['burn_in_period']
    scheduler = configs['scheduler']
    lower = configs['lower']
    upper = configs['upper']
    if training_type != 'at':
        kl_lambda = configs['kl_lambda']
    else:
        kl_lambda = None
    ## hyper-parameters for reweighting
    pm = configs['probabilistic_margin']
    gamma = configs['gamma']
    beta = configs['beta']
    ## PGD setting
    epsilon = configs['epsilon']/255
    alpha = configs['alpha']/255
    num_iters = configs['num_iters']
    #####################################################
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.checkpoint, exist_ok=True)
    
    tb_filename = os.path.join(args.checkpoint, 'logs')
    if os.path.exists(tb_filename):    shutil.rmtree(tb_filename)
    tb = SummaryWriter(log_dir=tb_filename)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='/root/mnt/datasets/data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='/root/mnt/datasets/data', train=False, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if model_type == 'ResNet':
        model = nn.DataParallel(resnet18(num_classes=num_classes).cuda())
    elif model_type == 'WRN':
        model = nn.DataParallel(WRN34_10().cuda())

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    xent = nn.CrossEntropyLoss().cuda()
    best_acc_nat = 0
    best_acc_adv = 0
    
    for epoch in range(epochs):
        training(epoch, model, train_dataloader, optimizer, alpha, epsilon, num_iters, 
                 gamma, beta, tb, num_classes, pm, lower, upper, burn_in_period, training_type, kl_lambda)
        test_acc_nat, test_acc_adv = evaluation(epoch, model, test_dataloader, alpha, epsilon, num_iters, lower, upper)
        tb.add_scalar('test/acc_nat', test_acc_nat, epoch)
        tb.add_scalar('test/acc_adv', test_acc_adv, epoch)

        is_best_nat = best_acc_nat < test_acc_nat
        is_best_adv = best_acc_adv < test_acc_adv
        best_acc_nat = max(best_acc_nat, test_acc_nat)
        best_acc_adv = max(best_acc_adv, test_acc_adv)
        save_checkpoint = {'state_dict': model.state_dict(),
                           'best_acc_nat': best_acc_nat,
                           'test_acc_nat': test_acc_nat,
                           'best_acc_adv': best_acc_adv,
                           'test_acc_adv': test_acc_adv,
                           'optimizer': optimizer.state_dict()}
        torch.save(save_checkpoint, os.path.join(args.checkpoint, 'model'))
        if is_best_nat and is_best_adv:
            print('Current model achieved the best accuracy, so saved as the best model.')
            torch.save(save_checkpoint, os.path.join(args.checkpoint, 'best_model'))
        adjust_learning_rate.step()
        