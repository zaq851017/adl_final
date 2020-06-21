import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel
import config
import ipdb
import sys 

class NERnet(nn.Module):
    def __init__(self):
        super(NERnet, self).__init__()
        self.bert_char = BertModel.from_pretrained(config.bert_char)
        #self.bert_word = BertModel.from_pretrained(config.bert)
        #self.embed_char = self.bert_char.embeddings.word_embeddings
        #self.embed_word = self.bert_word.embeddings.word_embeddings
        self.l1 = nn.Linear(768, 1)
        self.l2 = nn.Linear(768, 1)
        self.l3 = nn.Linear(768, 1)

    def forward(self, text, seg, mask):
        outputs = self.bert_char(text, token_type_ids=seg, attention_mask=mask)
        bound = max([(t==1).nonzero()[0] for t in seg])
        output1 = self.l1(outputs[1])
        output2 = self.l2(outputs[0].transpose(1, 0)[:bound].transpose(1, 0)) #start
        output3 = self.l3(outputs[0].transpose(1, 0)[:bound].transpose(1, 0)) #end

        return output1, output2, output3

class NERcnn(nn.Module):
    def __init__(self):
        super(NERcnn, self).__init__()
        self.bert_char = BertModel.from_pretrained(config.bert_char)
        #self.bert_word = BertModel.from_pretrained(config.bert)
        #self.embed_char = self.bert_char.embeddings.word_embeddings
        #self.embed_word = self.bert_word.embeddings.word_embeddings
        self.convs2 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768, 
                                    out_channels=1, 
                                    kernel_size=h))
                        for h in [1, 2, 3, 4, 5]
            ])
        self.convs3 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768, 
                                    out_channels=1, 
                                    kernel_size=h))
                        for h in [1, 2, 3, 4, 5]
            ])
        self.l1 = nn.Linear(768, 1)

    def forward(self, text, seg, mask):
        outputs = self.bert_char(text, token_type_ids=seg, attention_mask=mask)
        output1 = self.l1(outputs[1])
        bound = max([(t==1).nonzero()[0] for t in seg])
        out = outputs[0].transpose(1, 0)[:bound].permute(1,2,0)

        out2 = [conv(out) for conv in self.convs2]
        output2 = out2[0]
        for i in range(1, len(out2)):
            output2[:, :, :-i] = (output2[:, :, :-i] * i + out2[i]) / (i + 1)

        out3 = [conv(out) for conv in self.convs3]
        output3 = out3[0]
        for i in range(1, len(out3)):
            output3[:, :, :-i] = (output3[:, :, :-i] * i + out3[i]) / (i + 1)

        return output1, output2.transpose(1,2), output3.transpose(1,2)