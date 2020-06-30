import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, all_browsed, all_candidate, all_label):
        self.all_browsed = torch.from_numpy(all_browsed).long()
        self.all_label = torch.from_numpy(all_label).long()
        self.all_candidate = torch.from_numpy(all_candidate).long()
        self.len = all_browsed.shape[0]
        
    def __getitem__(self, index):
        return self.all_browsed[index], self.all_candidate[index], self.all_label[index]
    
    def __len__(self):
        return self.len


class NewsDataset(Dataset):
    def __init__(self, all_news_test):
        self.all_news_test = torch.from_numpy(all_news_test).long()
        self.len = all_news_test.shape[0]
    
    def __getitem__(self, index):
        return self.all_news_test[index]
    
    def __len__(self):
        return self.len


class UserDataset(Dataset):
    def __init__(self, user_browsed_test):
        self.user_browsed_test = torch.from_numpy(user_browsed_test)
        self.len = user_browsed_test.shape[0]
    
    def __getitem__(self, index):
        return self.user_browsed_test[index]
    
    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self, all_user_r_test, all_candidate_title_test):
        self.all_user_r_test = torch.from_numpy(all_user_r_test)
        self.all_candidate_title_test = torch.from_numpy(all_candidate_title_test)
        self.len = all_user_r_test.shape[0]
    
    def __getitem__(self, index):
        return self.all_user_r_test[index], self.all_candidate_title_test[index]
    
    def __len__(self):
        return self.len
