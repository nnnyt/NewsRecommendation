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
        self.category_embedding = nn.Embedding(config.category_num, config.category_embedding_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(config.subcategory_num, config.category_embedding_dim, padding_idx=0)
        
        self.title_cnn = nn.Conv2d(1, config.num_filters, (config.window_size, config.embedding_dim), padding=(1, 0))
        self.abstract_cnn = nn.Conv2d(1, config.num_filters, (config.window_size, config.embedding_dim), padding=(1, 0))
        self.title_attention = Attention(config.attention_dim, config.num_filters)
        self.abstract_attention = Attention(config.attention_dim, config.num_filters)

        self.category_dense = nn.Linear(config.category_embedding_dim, config.num_filters)
        self.subcategory_dense = nn.Linear(config.category_embedding_dim, config.num_filters)

        self.view_attention = Attention(config.attention_dim, config.num_filters)
    
    def forward(self, news):
        title = news[:, :self.config.max_title_len].to(device)
        abstract  = news[:, self.config.max_title_len : self.config.max_title_len + self.config.max_abstract_len].to(device)
        category = news[:, -2 : -1].to(device)
        subcategory = news[:, -1:].to(device)
        # title
        # batch_size, title_len, embedding_dim
        title_embedded = F.dropout(self.word_embedding(title), p=self.config.dropout)
        # batch_size, num_filters, title_len
        title_cnn = self.title_cnn(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, title_len, num_filters
        title_cnn = F.dropout(F.relu(title_cnn), p=self.config.dropout).transpose(1, 2)
        # batch_size, num_filters
        title_att = self.title_attention(title_cnn)

        # abstract
        # batch_size, abstract_len, embedding_dim
        abstract_embedded = F.dropout(self.word_embedding(abstract), p=self.config.dropout)
        # batch_size, num_filters, abstract_len
        abstract_cnn = self.abstract_cnn(abstract_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, abstract_len, num_filters
        abstract_cnn = F.dropout(F.relu(abstract_cnn), p=self.config.dropout).transpose(1, 2)
        # batch_size, num_filters
        abstract_att = self.abstract_attention(abstract_cnn)

        # category
        # batch_size, category_embedding_dim
        category_embedded = self.category_embedding(category).squeeze(dim=1)
        # batch_size, num_filters
        category_dense = F.relu(self.category_dense(category_embedded))

        # subcategory
        # batch_size, category_embedding_dim
        subcategory_embedded = self.subcategory_embedding(subcategory).squeeze(dim=1)
        # batch_size, num_filters
        subcategory_dense = F.relu(self.subcategory_dense(subcategory_embedded))

        # news_r
        # batch_size, 4, num_filters
        news_view = torch.stack([title_att, abstract_att, category_dense, subcategory_dense], dim=1)
        # batch_size, num_filters
        news_r = self.view_attention(news_view)

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


class NAML(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(NAML, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.news_encoder = NewsEncoder(config, pretrained_embedding)
        self.user_encoder = UserEncoder(config)
    
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
        return click_prob
    
    def get_news_r(self, title):
        return self.news_encoder(title)
    
    def get_user_r(self, browsed_news_r):
        return self.user_encoder(browsed_news_r.to(device))
    
    def test(self, user_r, candidate_news_r):
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()

