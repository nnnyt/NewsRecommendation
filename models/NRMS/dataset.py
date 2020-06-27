import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, all_browsed_title, all_candidate_title, all_label):
        self.all_browsed_title = torch.from_numpy(all_browsed_title).long()
        self.all_label = torch.from_numpy(all_label).long()
        self.all_candidate_title = torch.from_numpy(all_candidate_title).long()
        self.len = all_browsed_title.shape[0]
        
    def __getitem__(self, index):
        return self.all_browsed_title[index], self.all_candidate_title[index], 0
    
    def __len__(self):
        return self.len
