import numpy as np
from preprocess import preprocess_user_data, preprocess_test_user_data
from model import NRMS
from config import config
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset import MyDataset, NewsDataset, UserDataset, TestDataset
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def evaluate(impression_index, all_label_test, pred_label):
    from sklearn.metrics import roc_auc_score
    all_auc = []
    all_mrr = []
    all_ndcg5 = []
    all_ndcg10 = []
    for i in impression_index:
        begin = int(i[0])
        end = int(i[1])
        auc = roc_auc_score(all_label_test[begin:end], pred_label[begin:end])
        all_auc.append(auc)
        mrr = mrr_score(all_label_test[begin:end], pred_label[begin:end])
        all_mrr.append(mrr)
        if end - begin > 4:
            ndcg5 = ndcg_score(all_label_test[begin:end], pred_label[begin:end], 5)
            all_ndcg5.append(ndcg5)
            if end - begin > 9:
                ndcg10 = ndcg_score(all_label_test[begin:end], pred_label[begin:end], 10)
                all_ndcg10.append(ndcg10)
    return np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10)


def get_train_input(news_index, news_title):
    all_browsed_news, all_click, all_unclick, all_candidate, all_label = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
    print('preprocessing trainning input...')
    all_browsed_title = np.zeros((len(all_browsed_news), config.max_browsed, config.max_title_len), dtype='int32')
    # all_browsed_title = np.array([[ np.zeros(config.max_title_len, dtype='int32')for i in range(config.config.max_browsed)] for _ in all_browsed_news])
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < config.max_browsed:
                all_browsed_title[i][j] = news_title[news_index[news]]
            j += 1

    all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
    all_label = np.array(all_label)
    return all_browsed_title, all_candidate_title, all_label


def get_test_input(news_index_test, news_r_test):
    impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv')
    print('preprocessing testing input...')
    user_browsed_title_test = np.zeros((len(user_browsed_test), config.max_browsed, config.embedding_dim), dtype='float32')
    for i, user_browsed in enumerate(user_browsed_test):
        j = 0
        for news in user_browsed:
            if j < config.max_browsed:
                user_browsed_title_test[i][j] = news_r_test[news_index_test[news]]
            j += 1
    all_candidate_title_test = np.array([news_r_test[news_index_test[i[0]]] for i in all_candidate_test])
    all_label_test = np.array(all_label_test)
    return impression_index, user_index, user_browsed_title_test, all_user_test, all_candidate_title_test, all_label_test


def read_json(file='embedding_matrix.json'):
    with open(file, 'r') as f:
        embedding_matrix = json.load(f)
    return embedding_matrix


def get_embedding_matrix(word_index):
    if os.path.exists('embedding_matrix.json'):
        print('Load embedding matrix...')
        embedding_matrix = np.array(read_json())
    return embedding_matrix


def train(train_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, data in enumerate(train_iter):
            browsed, candidate, labels, true_labels = data
            # print(browsed.size(), candidate.size(), labels.size(), true_labels.size())
            optimizer.zero_grad()
            output = model(browsed, candidate)
            loss = torch.stack([x[0] for x in - F.log_softmax(output, dim=1)]).mean()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}'
                print(msg.format(i+1, loss.item()))


if __name__ == "__main__":
    news_index = np.load('news/news_index.npy', allow_pickle=True).item()
    news_index_test = np.load('news/news_index_test.npy', allow_pickle=True).item()
    word_index = np.load('news/word_index.npy', allow_pickle=True).item()
    news_title = np.load('news/news_title.npy', allow_pickle=True)
    news_title_test = np.load('news/news_title_test.npy', allow_pickle=True)
    all_browsed_title, all_candidate_title, all_label = get_train_input(news_index, news_title)

    pretrained_embedding = torch.from_numpy(get_embedding_matrix(word_index)).float()
    dataset = MyDataset(all_browsed_title, all_candidate_title, all_label)
    train_data = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    model = NRMS(config, pretrained_embedding).to(device)
    # print(model)
    train(train_data)

    model.eval()
    news_dataset = NewsDataset(news_title_test)
    news_dataloader = DataLoader(news_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print('get news representations...')
    news_r_test = []
    with torch.no_grad():
        for i, news_data in enumerate(news_dataloader):
            news_r = model.get_news_r(news_data)
            news_r = news_r.numpy()
            if i == 0:
                news_r_test = news_r
            else:
                news_r_test = np.concatenate((news_r_test, news_r), axis=0)

    impression_index, user_index, user_browsed_title_test, all_user_test, all_candidate_title_test, \
        all_label_test = get_test_input(news_index_test, news_r_test)
    user_dataset = UserDataset(user_browsed_title_test)
    user_dataloader = DataLoader(user_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print('get user representations...')
    user_r_test = []
    with torch.no_grad():
        for i, user_data in enumerate(user_dataloader):
            user_r = model.get_user_r(user_data)
            user_r = user_r.numpy()
            if i == 0:
                user_r_test = user_r
            else:
                user_r_test = np.concatenate((user_r_test, user_r), axis=0)
    all_user_r_test = np.zeros((len(all_user_test), config.embedding_dim), dtype='float32')
    for i, user in enumerate(all_user_test):
        all_user_r_test[i] = user_r_test[user_index[user]]
    test_dataset = TestDataset(all_user_r_test, all_candidate_title_test)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    print('test...')
    pred_label = []
    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            user, candidate = test_data
            pred = model.test(user, candidate)
            pred = pred.numpy()
            if i == 0:
                pred_label = pred
            else:
                pred_label = np.concatenate((pred_label, pred), axis=0)
    pred_label = np.array(pred_label).reshape(-1)
    print(pred_label.shape)
    all_label_test = np.array(all_label_test).reshape(-1)
    auc, mrr, ndcg5, ndcg10 = evaluate(impression_index, all_label_test, pred_label)
    print('auc: ', auc)
    print('mrr: ', mrr)
    print('ndcg5: ', ndcg5)
    print('ndcg10: ', ndcg10)
