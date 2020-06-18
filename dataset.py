import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class NERset(Dataset):
    def __init__(self,mode):
        root = 'textData_' + mode + '.pkl'
        with open(root, "rb") as f:
            self.data = pickle.load(f)
        self.train_or_test = mode

    def __getitem__(self, index):
        # sample[0]:name
        # sample[1]:text_tag
        # sample[2]:segments_tensor
        # sample[3]:index
        # sample[4]:index_bound
        # sample[5]:answer_able
        # sample[6]:start_label
        # sample[7]:end_label
        name = self.data[index]['file']
        text_tokens_tensors = self.data[index]['text']
        tag_tokens_tensors = self.data[index]['tag']
        r_index = self.data[index]['index']
        r_index_bound = self.data[index]['index_bound']
        text_decode = self.data[index]['text_decode']
        tag_decode = self.data[index]['tag_decode']
        filelen = self.data[index]['fileLen']
        if self.train_or_test=='train':
            start_label_ids = self.data[index]['val_start']
            end_label_ids = self.data[index]['val_end']
            answer_able = self.data[index]['able']
        elif self.train_or_test=='test':
            start_label_ids = None
            end_label_ids = None
            answer_able = None
        text_len = text_tokens_tensors.shape[0]
        tag_len = tag_tokens_tensors.shape[0]
        segments_tensor = torch.tensor([0]*text_len +[1]*tag_len)
        text_tag_tensors = torch.cat((text_tokens_tensors,tag_tokens_tensors))
        
        total_len = text_len + tag_len
        # max_tag_len = 16
        # max_text_len = 1003
        if total_len >=512:
            text_tokens_tensors = self.data[index]['text'][0:495]
            
            sep_tensor = torch.tensor([3])
            text_tag_tensors = torch.cat((text_tokens_tensors,sep_tensor))
            
            text_tag_tensors = torch.cat((text_tag_tensors,tag_tokens_tensors))
            text_len = 496
            segments_tensor = torch.tensor([0]*text_len +[1]*tag_len)
            
            
        
        if end_label_ids !=None and end_label_ids > text_len:
            answer_able = 0
            start_label_ids = -1
            end_label_ids = -1
            
        
        return (name,text_tag_tensors,segments_tensor,r_index,r_index_bound,answer_able,start_label_ids,end_label_ids, text_decode, tag_decode, filelen)
        

    def __len__(self):
        return len(self.data)
    def create_mini_batch(self,samples):
        # sample[0]:name
        # sample[1]:text_tag
        # sample[2]:segments_tensor
        # sample[3]:mask_tensor
        # sample[4]:index
        # sample[5]:index_bound
        # sample[6]:answer_able
        # sample[7]:start_label
        # sample[8]:end_label
       
        
        name = [s[0] for s in samples]
        text_tag_tensors = [s[1] for s in samples]
        segments_tensors = [s[2] for s in samples]
        index = [s[3] for s in samples]
        index_bound = [s[4] for s in samples]
        text_decode = [s[8] for s in samples]
        tag_decode = [s[9] for s in samples]
        filelen = [s[10] for s in samples]
        if self.train_or_test=='train':
            answerable = [s[5] for s in samples]
            start_tensors= [s[6] for s in samples]
            end_tensors= [s[7] for s in samples]
            answerable = torch.tensor([i for i in (answerable)])
            start_tensors  = torch.tensor([i for i in (start_tensors)])
            end_tensors = torch.tensor([i for i in (end_tensors)])
        else:
            answerable =None
            start_tensors = None
            end_tensors= None
        
       
        text_tag_tensors = pad_sequence(text_tag_tensors,batch_first=True)
        segments_tensors = pad_sequence(segments_tensors,batch_first=True)
        masks_tensors = torch.zeros(text_tag_tensors.shape,dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(text_tag_tensors != 0, 1)
        
        
            
        return name,text_tag_tensors,segments_tensors,masks_tensors,index,index_bound,answerable,start_tensors,end_tensors, text_decode, tag_decode, filelen

if __name__ == "__main__":
    ## usage of train
    dataset = NERset(mode='train')
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False,collate_fn=dataset.create_mini_batch)
    for i,data in enumerate(dataloader):
        if data[1].shape[1] >500:
            print(data)
    ## usage of test
    dataset = NERset(mode='test')
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False,collate_fn=dataset.create_mini_batch)
    for i,data in enumerate(dataloader):
        if data[1].shape[1] >500:
            print(data)
