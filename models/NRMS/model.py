import torch
import torch.nn as nn
import torch.nn.functional as F
from SelfAttention import SelfAttention
from attention import Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.multi_head_self_attention = SelfAttention(config.nb_head, config.embedding_dim)
        self.attention = Attention(config.attention_dim, config.embedding_dim)
        if pretrained_embedding is None:
            self.word_embedding = nn.Embedding(config.word_num, config.embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding,freeze=False)
    
    def forward(self, title):
        # batch_size, title_len, embedding_dim
        title_embedded = F.dropout(self.word_embedding(title.to(device)), p=self.config.dropout)
        # batch_size, title_len, embedding_dim
        title_selfatt = F.dropout(self.multi_head_self_attention(title_embedded), p=self.config.dropout)
        # batch_size, embedding_dim
        news_r = self.attention(title_selfatt)
        return news_r


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.multi_head_self_attention = SelfAttention(config.nb_head, config.embedding_dim)
        self.attention = Attention(config.attention_dim, config.embedding_dim)
    
    def forward(self, news):
        # batch_size, browsed_num, embedding_dim
        self_att = F.dropout(self.multi_head_self_attention(news), self.config.dropout)
        # batch_size, embedding_dim
        user_r = self.attention(self_att)
        return user_r


class NRMS(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.news_encoder = NewsEncoder(config, pretrained_embedding)
        self.user_encoder = UserEncoder(config)
    
    def forward(self, browsed_news, candidate_news):
        # browsed_num, batch_size, title_len
        browsed_news = browsed_news.transpose(0,1)
        # 1+K, batch_size, title_len
        candidate_news = candidate_news.transpose(0,1)
        # 1+K, batch_size, embedding_dim
        candidate_news_r = torch.stack([self.news_encoder(x) for x in candidate_news])
        # batch_size, browsed_num, embedding_dim
        browsed_news_r = torch.stack([self.news_encoder(x) for x in browsed_news], dim=1)
        # batch_size, embedding_dim
        user_r = self.user_encoder(browsed_news_r)
        # batch_size, 1+K
        click_prob = torch.stack([torch.bmm(user_r.unsqueeze(dim=1), x.unsqueeze(dim=2)).flatten()
                    for x in candidate_news_r], dim=1)
        return click_prob
    
    def get_news_r(self, title):
        return self.news_encoder(title)
    
    def get_user_r(self, browsed_news_r):
        return self.user_encoder(browsed_news_r.to(device))
    
    def test(self, user_r, candidate_news_r):
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()

