import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn as nn
from dataset import NERset
from backbone import NERnet
import config
import ipdb
import sys 
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

def train(args):
    root = str(datetime.now().strftime('%m%d_%H:%M'))
    log = open(os.path.join('log', root + ".log"), "w")
    root = os.path.join('log', 'model', root)
    try:
        os.mkdir(root)
    except:
        pass
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    trainset = NERset(mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn = trainset.create_mini_batch)
    #validset = NERset(mode='test')
    #validloader = DataLoader(validset, batch_size=1, shuffle=False, collate_fn = trainset.create_mini_batch)

    net = NERnet()
    if multi_gpus:
        net = nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    threshold = torch.tensor([args.threshold]).to(device)
    optimizer = optim.AdamW(net.parameters(),lr = args.lr, weight_decay=args.weight_decay)
    criterion1 = nn.BCEWithLogitsLoss().to(device)
    criterion2 = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    for epoch in range(20):
        print("Epoch: {}".format(epoch))
        log.writelines("Epoch: {}\n".format(epoch))
        train_loss = 0
        valid_loss = 0
        net.train()
        for batch, data in enumerate(trainloader):
            sys.stdout.write("    Train Batch: {}/{}\r".format(batch, len(trainloader)))
            sys.stdout.flush()
            net.zero_grad()
            name, text, seg, mask, index, index_bound, answerable, start, end = data
            text = text.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            output1, output2, output3 = net(text, seg, mask)
            #ipdb.set_trace()
            #answerable = torch.tensor(answerable.clone().detach(), dtype=torch.float).to(device) 
            answerable = answerable.float().to(device)
            start = start.to(device)
            end = end.to(device)
            loss1 = criterion1(output1.view(-1, 1), answerable.view(-1, 1))
            loss2 = criterion2(output2.squeeze(-1), start)
            loss3 = criterion2(output3.squeeze(-1), end)
            #loss = (loss1 + loss2 + loss3) / 3
            if loss2 == 0:
                loss = loss1
            else:
                loss = (loss1 + loss2 + loss3) / 3
            loss.backward()
            optimizer.step()
            train_loss += loss
            sys.stdout.write("    Train Batch: {}/{}, Batch Loss: {:.6f}({:.6f}, {:.6f}, {:.6f})\r".format(batch, len(trainloader), loss.item(), loss1.item(), loss2.item(), loss3.item()))
            sys.stdout.flush() 

        print("\n    Train loss: {}".format(train_loss/len(trainloader)))
        log.writelines("    Train loss: {}\n".format(train_loss/len(trainloader)))
        savepath = os.path.join(root, str(epoch)+'.pth')
        print("Save model: {}".format(savepath))
        log.writelines("Save model: {}".format(savepath))
        torch.save(net.state_dict(), savepath)   

if __name__ == '__main__':
    args = config.args
    train(args)