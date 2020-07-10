import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class KCNN(nn.Module):
    def __init__(self, config, pretrained_embedding=None, pretrained_entity_embedding=None):
        super(KCNN, self).__init__()
        self.config = config
        if pretrained_embedding is None:
            self.word_embedding = nn.Embedding(config.word_num, config.embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding,freeze=False)
        if pretrained_entity_embedding is None:
            self.entity_embedding = nn.Embedding(config.entity_num, config.entity_embedding_dim, padding_idx=0)
        else:
            self.entity_embedding = nn.Embedding.from_pretrained(pretrained_entity_embedding,freeze=False)
        self.transform = nn.Linear(config.entity_embedding_dim, config.embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(2, config.num_filters, (k, config.embedding_dim)) for k in config.window_size])
        self.att = Attention(config.attention_dim, config.num_filters)

    def forward(self, news):
        title = news[:, :self.config.max_title_len].to(device)
        entity = news[:, self.config.max_title_len : ].to(device)
        # batch_size, title_len, embedding_dim
        title_embedded = F.dropout(self.word_embedding(title), p=self.config.dropout)
        # batch_size, title_len, entity_embedding_len
        entity_embedded = F.dropout(self.entity_embedding(entity), p=self.config.dropout)
        # batch_size, title_len, embedding_dim
        entity_transformed = torch.tanh(self.transform(entity_embedded))
        # batch_size, 2, title_len, embedding_dim
        embedding = torch.stack([title_embedded, entity_transformed], dim=1)
        pooled = []
        for conv in self.conv:
            # batch_size, num_filters, title_len - window_size + 1
            cnn = F.relu(conv(embedding).squeeze(dim=3))
            # batch_size, num_filters
            # cnn_pooled = F.max_pool1d(cnn, cnn.size(2)).squeeze(dim=2)
            cnn_pooled = self.att(cnn.transpose(1, 2))
            pooled.append(cnn_pooled)
        # batch_size, num_filters * len(window_size)
        new_r = torch.cat(pooled, dim=1)
        return new_r


class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.dnn = nn.Sequential(
            nn.Linear(len(config.window_size) * config.num_filters * 2, 64),
            nn.Linear(64, 1))

    def forward(self, news, candidate):
        # candidate: batch_size, num_filters * len(window_size)
        # news: batch_size, browsed_len, num_filters * len(window_size)
        # batch_size, browsed_len, num_filters * len(window_size) * 2
        concat_news = torch.cat([news, candidate.expand(self.config.max_browsed, -1, -1).transpose(0, 1)], dim=2)
        # batch_size, browsed_len
        att_weight = F.softmax(self.dnn(concat_news).squeeze(dim=2), dim=1)
        # batch_size, num_filters * len(window_size)
        user_r = torch.bmm(att_weight.unsqueeze(dim=1), news).squeeze(dim=1)
        return user_r


class DKN(nn.Module):
    def __init__(self, config, pretrained_embedding=None, pretrained_entity_embedding=None):
        super(DKN, self).__init__()
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.pretrained_entity_embedding = pretrained_entity_embedding
        self.kcnn = KCNN(config, pretrained_embedding, pretrained_entity_embedding)
        self.user_encoder = UserEncoder(config)
    
    def forward(self, browsed_news, candidate_news):
        # browsed_num, batch_size, title_len * 2
        browsed_news = browsed_news.transpose(0, 1)
        # 1+K, batch_size, title_len+abstract_len+2
        candidate_news = candidate_news.transpose(0, 1)
        # batch_size, browsed_num, num_filters * len(window_size)
        browsed_news_r = torch.stack([self.kcnn(x) for x in browsed_news], dim=1)
        # 1+K, batch_size, num_filters * len(window_size)
        candidate_news_r = torch.stack([self.kcnn(x) for x in candidate_news])
        # 1+K, batch_size, num_filters * len(window_size)
        user_r = torch.stack([self.user_encoder(browsed_news_r, x) for x in candidate_news_r])
        # batch_size, 1
        # click_prob = torch.bmm(user_r.unsqueeze(dim=1), candidate_news_r.unsqueeze(dim=2)).flatten()
        # use nagetive sampling
        # batch_size, 1+K
        click_prob = torch.stack([
            torch.bmm(x.unsqueeze(dim=1), y.unsqueeze(dim=2)).flatten() for (x, y) in zip(user_r, candidate_news_r)
        ], dim=1)
        # print(click_prob.size())
        return click_prob
    
    def get_news_r(self, news):
        return self.kcnn(news)
    
    
    def test(self, browsed_news_r, candidate_news_r):
        user_r = self.user_encoder(browsed_news_r.to(device), candidate_news_r.to(device))
        return torch.bmm(user_r.to(device).unsqueeze(dim=1), candidate_news_r.to(device).unsqueeze(dim=2)).flatten()



