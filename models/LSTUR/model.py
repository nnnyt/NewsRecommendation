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
        self.title_attention = Attention(config.attention_dim, config.num_filters)
        
    
    def forward(self, news):
        # batch_size, max_title_len
        title = news[:, :self.config.max_title_len].to(device)
        # batch_size, 1
        category = news[:, -2 : -1].to(device)
        # batch_size, 1
        subcategory = news[:, -1 :].to(device)

        # title
        # batch_size, title_len, embedding_dim
        title_embedded = F.dropout(self.word_embedding(title), p=self.config.dropout)
        # batch_size, num_filters, title_len
        title_cnn = self.title_cnn(title_embedded.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, title_len, num_filters
        title_cnn = F.dropout(F.relu(title_cnn), p=self.config.dropout).transpose(1, 2)
        # batch_size, num_filters
        title_att = self.title_attention(title_cnn)

        # category
        # batch_size, category_embedding_dim
        category_embedded = self.category_embedding(category).squeeze(dim=1)
        # subcategory
        # batch_size, category_embedding_dim
        subcategory_embedded = self.subcategory_embedding(subcategory).squeeze(dim=1)

        # batch_size, num_filters + category_embedding_dim * 2
        news_r = torch.cat([title_att, category_embedded, subcategory_embedded], dim=1)

        return news_r


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        assert (config.num_filters + config.category_embedding_dim * 2) / 2 % 1 == 0
        self.user_dim = config.num_filters + config.category_embedding_dim * 2
        if config.model_type == 'ini':
            weight = torch.zeros(config.user_num, self.user_dim)
            self.gru = nn.GRU(self.user_dim, self.user_dim, batch_first=True)
            self.user_embedding = nn.Embedding(config.user_num, self.user_dim, padding_idx=0)
        else:
            weight = torch.zeros(config.user_num, int(self.user_dim / 2))
            self.gru = nn.GRU(self.user_dim, int(self.user_dim / 2), batch_first=True)
            self.user_embedding = nn.Embedding(config.user_num, int(self.user_dim / 2), _weight=weight)
    
    def forward(self, news, user, browsed_len):
        # batch_size, user_dim or user_dim/2
        user_long_r = F.dropout2d(self.user_embedding(user).squeeze(dim=1).unsqueeze(dim=0), 
                                p=self.config.masking_probability, training=True).squeeze(dim=0)
        # batch_size, user_dim
        if self.config.model_type == 'ini':
            # packed_news = nn.utils.rnn.pack_padded_sequence(news, browsed_len,
                                                            # batch_first=True, enforce_sorted=False)
            _, user_r = self.gru(news, user_long_r.unsqueeze(dim=0))
            user_r = user_r.squeeze(dim=0)
        else:
            # packed_news = nn.utils.rnn.pack_padded_sequence(news, browsed_len, 
                                                            # batch_first=True, enforce_sorted=False)
            _, user_news_r = self.gru(news)
            user_r = torch.cat((user_news_r.squeeze(dim=0), user_long_r), dim=1)
        return user_r


class LSTUR(nn.Module):
    def __init__(self, config, pretrained_embedding=None):
        super(LSTUR, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.news_encoder = NewsEncoder(config, pretrained_embedding)
        self.user_encoder = UserEncoder(config)
    
    def forward(self, browsed_news, candidate_news, user_id, browsed_len):
        # browsed_num, batch_size, title_len+abstract_len+2
        browsed_news = browsed_news.transpose(0, 1)
        # 1+K, batch_size, title_len+abstract_len+2
        candidate_news = candidate_news.transpose(0, 1)
        # 1+K, batch_size, num_filters
        candidate_news_r = torch.stack([self.news_encoder(x) for x in candidate_news])
        # batch_size, browsed_num, num_filters
        browsed_news_r = torch.stack([self.news_encoder(x) for x in browsed_news], dim=1)
        # batch_size, num_filters
        user_r = self.user_encoder(browsed_news_r, user_id.to(device), browsed_len.to(device))
        # batch_size, 1+K
        click_prob = torch.stack([torch.bmm(user_r.unsqueeze(dim=1), x.unsqueeze(dim=2)).flatten()
                    for x in candidate_news_r], dim=1)
        return click_prob
    
    def get_news_r(self, news):
        return self.news_encoder(news)
    
    def get_user_r(self, browsed_news_r, user_id, browsed_len):
        return self.user_encoder(browsed_news_r.to(device), user_id.to(device), browsed_len.to(device))
    
    def test(self, user_r, candidate_news_r):
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()

