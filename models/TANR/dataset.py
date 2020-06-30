import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, all_browsed_title, all_candidate_title, all_label, news_category):
        self.all_browsed_title = torch.from_numpy(all_browsed_title).long()
        self.all_label = torch.from_numpy(all_label).long()
        self.all_candidate_title = torch.from_numpy(all_candidate_title).long()
        self.news_category = torch.from_numpy(news_category).long()
        self.len = all_browsed_title.shape[0]
        
    def __getitem__(self, index):
        return self.all_browsed_title[index], self.all_candidate_title[index], \
                    self.all_label[index], self.news_category[index]
    
    def __len__(self):
        return self.len


class NewsDataset(Dataset):
    def __init__(self, news_title_test):
        self.news_title_test = torch.from_numpy(news_title_test).long()
        self.len = news_title_test.shape[0]
    
    def __getitem__(self, index):
        return self.news_title_test[index]
    
    def __len__(self):
        return self.len


class UserDataset(Dataset):
    def __init__(self, user_browsed_title_test):
        self.user_browsed_title_test = torch.from_numpy(user_browsed_title_test)
        self.len = user_browsed_title_test.shape[0]
    
    def __getitem__(self, index):
        return self.user_browsed_title_test[index]
    
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
