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
from backbone import NERnet, NERcnn, dualBERT
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
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        dataset = NERset(mode=args.mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = dataset.create_mini_batch)
        threshold = torch.tensor([args.threshold]).to(device)
        if args.backbone == 'cnn':
            net = NERcnn()
        elif args.backbone == 'dualbert':
            net = dualBERT()
        else:
            net = NERnet()
        net.to(device)
        net.load_state_dict(torch.load(args.model))
        net.eval()
        filename = set()
        answer = [['ID', 'Prediction']]
        batchans = []
        lastname = ''
        for batch in tqdm(dataloader):
            name, text, word, seg, mask, index, index_bound, answerable, start, end, text_decode, tag_decode, filelen = batch
            #name = name[0][0:-9]
            if tag_decode[0] == '質問箇所　ＴＥＬ／ＦＡＸ':
                tag_decode[0] = '質問箇所TEL/FAX'
            if tag_decode[0] == '需要場所（住所）':
                tag_decode[0] = '需要場所(住所)'
            if tag_decode[0] == '質問箇所所属／担当者':
                tag_decode[0] = '質問箇所所属/担当者'
            if tag_decode[0] == '資格申請送付先　部署／担当者名':
                tag_decode[0] = '資格申請送付先部署/担当者名'
            if tag_decode[0] == '入札書送付先　部署／担当者名':
                tag_decode[0] = '入札書送付先部署/担当者名'
            if name[0][0:-9] not in filename:
                for j, a in enumerate(batchans):
                    if len(filename) > 0:
                        if a == 'NONE':
                            answer.append([lastname+"-"+str(j+1), a])
                        else:
                            a.sort(key = lambda tup: tup[1])
                            ans = ''
                            for t in a:
                                ans += t[0] + " "
                            answer.append([lastname+"-"+str(j+1), ans[:-1]])
                filename.add(name[0][0:-9])
                batchans = ['NONE'] * filelen[0]
            if args.mode =="dev":
                df = pd.read_excel(os.path.join(args.dev_path, 'ca_data', name[0]))
            else:
                df = pd.read_excel(os.path.join(args.test_path, 'ca_data', name[0]))
            
            new_index = []
            for ii in index[0]:
                new_index.append(df[df['Index']==ii].index.values[0])
            lastname = name[0][0:-9]
            text = text.to(device)
            word = word.to(device)
            seg = seg.to(device)
            mask = mask.to(device)
            if args.backbone == 'dualbert':
                output1, output2, output3 = net(text, word, seg, mask)
            else:
                output1, output2, output3 = net(text, seg, mask)
            answerable = (torch.sigmoid(output1)>threshold).float().view(-1)
            start = torch.topk(output2.squeeze(-1), k=1, dim=1)
            end = torch.topk(output3.squeeze(-1), k=1, dim=1)
            for j, a in enumerate(answerable):
                if a == 1:
                    s = start[1][j]
                    e = end[1][j]
                    if s >= e:
                        continue
                    for k, rr in enumerate(index_bound[j]):
                        if s in range(rr[0], rr[1]) or e - 1 in range(rr[0], rr[1]):
                            ans = tag_decode[j] + ':' + text_decode[j][max(s-1, rr[0]-1) : min(e-1,rr[1]-1)]
                            if batchans[new_index[k]] == 'NONE':
                                batchans[new_index[k]] = [(ans, s)]
                            else:
                                #batchans[new_index[k]] += " " + ans
                                batchans[new_index[k]].append((ans, s))
        for j, a in enumerate(batchans):
            if a == 'NONE':
                answer.append([lastname+"-"+str(j+1), a])
            else:
                a.sort(key = lambda tup: tup[1])
                ans = ''
                for t in a:
                    ans += t[0] + " "
                answer.append([lastname+"-"+str(j+1), ans[:-1]])
        with open("tmp_file", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(answer)
        with open("tmp_file") as csvfile:
            rows = csv.reader(csvfile)
            csvfile=open(args.output_file, 'w')
            writer = csv.writer(csvfile)
            writer.writerow(['ID','Prediction'])
            if args.mode =="dev":
                path = config.dev_path
            elif args.mode=="test":
                path = config.test_path
            list_files = os.listdir(path)
            list_files.sort()
            ll=[]
            for files in list_files:
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