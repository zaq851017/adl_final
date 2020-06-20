import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from dataset import NERset
from backbone import NERnet
import config
import ipdb
import sys 
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd

def predict(args):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        dataset = NERset(mode=args.mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = dataset.create_mini_batch)
        threshold = torch.tensor([args.threshold]).to(device)
        net = NERnet().to(device)
        net.load_state_dict(torch.load(args.model))
        net.eval()
        filename = set()
        answer = [['ID', 'Prediction']]
        batchans = []
        lastname = ''
        for batch in tqdm(dataloader):
            name, text, seg, mask, index, index_bound, answerable, start, end, text_decode, tag_decode, filelen = batch
            #name = name[0][0:-9]
            if name[0][0:-9] not in filename:
                for j, a in enumerate(batchans):
                    if len(filename) > 0:
                        answer.append([lastname+"-"+str(j+1), a])
                filename.add(name[0][0:-9])
                batchans = ['NONE'] * filelen[0]
            lastname = name[0][0:-9]
            text = text.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            output1, output2, output3 = net(text, seg, mask)
            answerable = (torch.sigmoid(output1)>threshold).float().view(-1)
            start = torch.topk(output2.squeeze(-1), k=1, dim=1)
            end = torch.topk(output3.squeeze(-1), k=1, dim=1)
            for j, a in enumerate(answerable):
                if a == 1:
                    s = start[1][j]
                    e = end[1][j]
                    for k, rr in enumerate(index_bound[j]):
                        if s in range(rr[0], rr[1]) or e - 1 in range(rr[0], rr[1]):
                            ans = tag_decode[j] + ':' + text_decode[j][max(s-1, rr[0]-1) : min(e-1,rr[1]-1)]
                            if batchans[index[j][k] - 1] == 'NONE':
                                batchans[index[j][k] - 1] = ans
                            else:
                                batchans[index[j][k] - 1] += " " + ans
        for j, a in enumerate(batchans):
                if len(filename) > 0:
                    answer.append([lastname+"-"+str(j+1), a])
        with open("tmp_file", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(answer)
        with open("tmp_file") as csvfile:
            rows = csv.reader(csvfile)
            csvfile=open(args.output_file, 'w')
            writer = csv.writer(csvfile)
            writer.writerow(['ID','Prediction'])
            test_path = "./release/test/ca_data/"
            test_files = os.listdir(test_path)
            test_files.sort()
            path = "./release/test/ca_data/"
            ll=[]
            for files in test_files:
                df=pd.read_excel(path + files)
                index_list = df['Index'].tolist()
                for i in index_list:
                    ll.append(files[0:-9]+"-"+str(i))
            print(len(ll))
            for i,row in enumerate(rows):
                if i==0:
                    continue
                else:
                    #print(ll[i-1])
                    row[0] = ll[i-1]
                    writer.writerow(row)
if __name__ == '__main__':
    args = config.args
    predict(args)