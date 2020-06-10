import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence

class NERset(Dataset):
    def __init__(self):
        with open("textData.pkl", "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        name = self.data[index]['name']
        char_tokens_tensors = self.data[index]['char_input']
        word_tokens_tensors = self.data[index]['word_input']
        label_ids = self.data[index]['char_tag']
        segments_tensor = torch.tensor([0]*char_tokens_tensors.shape[0])
        return (name,char_tokens_tensors,word_tokens_tensors,segments_tensor,label_ids)

    def __len__(self):
        return len(self.data)
    def create_mini_batch(self,samples):
        # sample[0]:name
        # sample[1]:char
        # sample[2]:word
        # sample[3]:segments
        # sample[4]:label
        
        name = [s[0] for s in samples]
        char_tokens_tensors = [s[1] for s in samples]
        word_tokens_tensors = [s[2] for s in samples]
        segments_tensors = [s[3] for s in samples]
        label_tensors= [s[4] for s in samples]
        
        before_pad_length=[]
        for i in range(len(char_tokens_tensors)):
            before_pad_length.append(char_tokens_tensors[i].shape[0])
        
        char_tokens_tensors = pad_sequence(char_tokens_tensors,batch_first=True)
        word_tokens_tensors = pad_sequence(word_tokens_tensors,batch_first=True)
        segments_tensors = pad_sequence(segments_tensors,batch_first=True)
        max_pad_length = char_tokens_tensors[0].shape[0]
        masks_tensors = torch.zeros(char_tokens_tensors.shape,dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(char_tokens_tensors != 0, 1)
        other_label = torch.tensor([1]*1+[0]*20,dtype=torch.float)
        other_label = torch.unsqueeze(other_label,0)
        
        for i in range(len(before_pad_length)):
            for j in range(max_pad_length-before_pad_length[i]):
                label_tensors[i] = torch.cat((label_tensors[i],other_label),0)
        r_label_tensors=torch.stack([i for i in (label_tensors)])
            
        return name,char_tokens_tensors,word_tokens_tensors,masks_tensors,r_label_tensors

if __name__ == "__main__":
    dataset = NERset()
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False,collate_fn=dataset.create_mini_batch)
    print(len(dataloader))
    for i,data in enumerate(dataloader):
        print(data)
