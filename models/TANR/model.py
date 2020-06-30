import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_embedding is None:
            self.word_embedding = nn.Embedding(config.word_num, config.embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding,freeze=False)
        
        self.title_cnn = nn.Conv2d(1, config.num_filters, (config.window_size, config.embedding_dim), padding=(1, 0))
        self.title_attention = Attention(config.attention_dim, config.num_filters)
    
    def forward(self, title):
        # title
        # batch_size, title_len, embedding_dim
        title_embedded = F.dropout(self.word_embedding(title.to(device)), p=self.config.dropout)
        # batch_size, num_filters, title_len
        title_cnn = self.title_cnn(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, title_len, num_filters
        title_cnn = F.dropout(F.relu(title_cnn), p=self.config.dropout).transpose(1, 2)
        # batch_size, num_filters
        news_r = self.title_attention(title_cnn)

        return news_r


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.attention = Attention(config.attention_dim, config.num_filters)
    
    def forward(self, news):
        # batch_size, num_filters
        user_r = self.attention(news)
        return user_r


class TANR(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(TANR, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.news_encoder = NewsEncoder(config, pretrained_embedding)
        self.user_encoder = UserEncoder(config)
        self.category_dense = nn.Linear(config.num_filters, config.category_num)
    
    def forward(self, browsed_news, candidate_news):
        # browsed_num, batch_size, title_len+abstract_len+2
        browsed_news = browsed_news.transpose(0, 1)
        # 1+K, batch_size, title_len+abstract_len+2
        candidate_news = candidate_news.transpose(0, 1)
        # 1+K, batch_size, num_filters
        candidate_news_r = torch.stack([self.news_encoder(x) for x in candidate_news])
        # batch_size, browsed_num, num_filters
        browsed_news_r = torch.stack([self.news_encoder(x) for x in browsed_news], dim=1)
        # batch_size, num_filters
        user_r = self.user_encoder(browsed_news_r)
        # batch_size, 1+K
        click_prob = torch.stack([torch.bmm(user_r.unsqueeze(dim=1), x.unsqueeze(dim=2)).flatten()
                    for x in candidate_news_r], dim=1)
        # batch_size, 1+K+browsed_num, num_filters
        all_news_r = torch.cat((candidate_news_r.transpose(0,1), browsed_news_r), dim=1)
        # batch_size, 1+K+browsed_num, category_num
        category_pred = self.category_dense(all_news_r)
        return click_prob, category_pred
    
    def get_news_r(self, title):
        return self.news_encoder(title)
    
    def get_user_r(self, browsed_news_r):
        return self.user_encoder(browsed_news_r.to(device))
    
    def test(self, user_r, candidate_news_r):
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()

