import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, all_browsed, all_candidate, all_user, all_browsed_len, all_label):
        self.all_browsed = torch.from_numpy(all_browsed).long()
        self.all_label = torch.from_numpy(all_label).long()
        self.all_candidate = torch.from_numpy(all_candidate).long()
        self.all_user = torch.from_numpy(all_user).long()
        self.all_browsed_len = torch.from_numpy(all_browsed_len).long()
        self.len = all_browsed.shape[0]
        
    def __getitem__(self, index):
        return self.all_browsed[index], self.all_candidate[index], \
                    self.all_user[index], self.all_browsed_len[index], self.all_label[index]
    
    def __len__(self):
        return self.len


class NewsDataset(Dataset):
    def __init__(self, news_test):
        self.news_test = torch.from_numpy(news_test).long()
        self.len = news_test.shape[0]
    
    def __getitem__(self, index):
        return self.news_test[index]
    
    def __len__(self):
        return self.len


class UserDataset(Dataset):
    def __init__(self, user_browsed, user_id, browsed_len):
        self.user_browsed = torch.from_numpy(user_browsed)
        self.user_id = torch.from_numpy(user_id)
        self.browsed_len = torch.from_numpy(browsed_len)
        self.len = user_browsed.shape[0]
    
    def __getitem__(self, index):
        return self.user_browsed[index], self.user_id[index], self.browsed_len[index]
    
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
