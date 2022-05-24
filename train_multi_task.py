# -*- coding:utf-8 -*-

import os
# import apex
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
# from apex import amp
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset_multi_task import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.conf.constant import Constant
from qdnet_classifier.classifier_multi_task import MultiLabelModel


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        optimizer.zero_grad()
        data, target = data.to(device), target#.to(device)
        output = model(data)
        loss = model.get_loss( output, target )

        if not config["use_amp"]:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if int(config["image_size"]) in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss



def val_epoch(model, loader, get_output=False):

    model.eval()
    val_loss = []
    color_acc = []
    action_acc = []
    color_action_acc1 = []
    color_action_acc3 = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target#.to(device)
            # probs = torch.zeros((data.shape[0], int(config["out_dim"]))).to(device)
            output = model(data)
            #probs = F.softmax(output,dim =1)
            #probs = probs.cpu().detach().numpy()
            #output_list += list( probs.argmax(1) )
            #target_list += list( target.cpu().detach().numpy() )
            # acc1, acc3, {'color': acc1_color, 'action': acc1_action}
            acc1, acc3, acc_dict = model.get_accuracy( output, target, topk=(1, 3) )

            color_acc.append( acc_dict['color'].detach().cpu() )
            action_acc.append( acc_dict['action'].detach().cpu() )
            color_action_acc1.append( acc1.detach().cpu() )
            color_action_acc3.append( acc3.detach().cpu() )

            loss = model.get_loss( output, target )
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    color_acc = np.mean(color_acc)
    action_acc = np.mean(action_acc)
    color_action_acc1 = np.mean(color_action_acc1)
    color_action_acc3 = np.mean(color_action_acc3)

    return val_loss, color_acc, action_acc, color_action_acc1, color_action_acc3


def run(fold, df, transforms_train, transforms_val):

    df_train = df[df['fold'] != fold]
    df_valid = df[df['fold'] == fold]

    dataset_train = QDDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = QDDataset(df_valid, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(config["batch_size"]), sampler=RandomSampler(dataset_train), num_workers=int(config["num_workers"]))
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]))

    model = ModelClass(
        config["enet_type"],
        config["out_dim1"],
        config["out_dim2"],
        pretrained = config["pretrained"] )
    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    acc_max = 0.  
    model_file  = os.path.join(config["model_dir"], f'best_fold{fold}.pth')
    model_file3 = os.path.join(config["model_dir"], f'final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=float(config["init_lr"]))
    if config["use_amp"]: 
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)
    #scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["n_epochs"]) - 1) 
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(config["n_epochs"]) - 1) 
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, int(config["n_epochs"]) + 1): 
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, color_acc, action_acc, acc1, acc3 = val_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, color_acc: {(color_acc):.4f}, action_acc: {(action_acc):.6f}.'
        print(content)  
        with open(os.path.join(config["log_dir"], f'log.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() 
            
        if acc1 > acc_max:
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc1))
            torch.save(model.state_dict(), model_file)
            acc_max = acc1

    torch.save(model.state_dict(), model_file3)


def main():

    df, df_test = get_df( config["data_dir"]  )

    transforms_train, transforms_val = get_transforms(config["image_size"])  

    folds = [int(i) for i in config["fold"].split(',')]  
    for fold in folds:
        run(fold, df, transforms_train, transforms_val)


if __name__ == '__main__':

    os.makedirs(config["model_dir"], exist_ok=True)  
    os.makedirs(config["log_dir"], exist_ok=True)    
    os.environ['CUDA_VISIBLE_DEVICES'] = config["CUDA_VISIBLE_DEVICES"]

    if config["enet_type"] in Constant.RESNET_LIST:
         ModelClass = MultiLabelModel
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()

