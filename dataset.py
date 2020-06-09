import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import ipdb

class NERset(Dataset):
    def __init__(self):
        with open("textData.pkl", "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
       return self.data[index] 

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    dataset = NERset()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    for data in dataloader:
        ipdb.set_trace()
