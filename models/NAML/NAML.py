import numpy as np
from preprocess import preprocess_user_data, preprocess_test_user_data
from model import NAML
from config import Config
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


def get_train_input(news_category, news_subcategory, news_abstract, news_title, news_index):
    all_browsed_news, all_click, all_unclick, all_candidate, all_label = preprocess_user_data('../../data/MINDsmall_train/behaviors.tsv')
    
    all_browsed_title = np.zeros((len(all_browsed_news), Config.max_browsed, Config.max_title_len), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < Config.max_browsed:
                all_browsed_title[i][j] = news_title[news_index[news]]
            j += 1

    all_browsed_abstract = np.zeros((len(all_browsed_news), Config.max_browsed, Config.max_abstract_len), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < Config.max_browsed:
                all_browsed_abstract[i][j] = news_abstract[news_index[news]]
            j += 1

    all_browsed_category = np.zeros((len(all_browsed_news), Config.max_browsed, 1), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < Config.max_browsed:
                all_browsed_category[i][j] = news_category[news_index[news]]
            j += 1

    all_browsed_subcategory = np.zeros((len(all_browsed_news), Config.max_browsed, 1), dtype='int32')
    for i, user_browsed in enumerate(all_browsed_news):
        j = 0
        for news in user_browsed:
            if j < Config.max_browsed:
                all_browsed_subcategory[i][j] = news_subcategory[news_index[news]]
            j += 1

    all_browsed = np.concatenate((all_browsed_title, all_browsed_abstract, all_browsed_category, all_browsed_subcategory), axis=-1)
    all_candidate_title = np.array([[ news_title[news_index[j]] for j in i] for i in all_candidate])
    all_candidate_abstract = np.array([[ news_abstract[news_index[j]] for j in i] for i in all_candidate])
    all_candidate_category = np.array([[ news_category[news_index[j]] for j in i] for i in all_candidate])
    all_candidate_subcategory = np.array([[ news_subcategory[news_index[j]] for j in i] for i in all_candidate])
    all_candidate = np.concatenate((all_candidate_title, all_candidate_abstract, all_candidate_category, all_candidate_subcategory), axis=-1)
    all_label = np.array(all_label)
    return all_browsed, all_candidate, all_label


def get_test_input(news_r_test, news_index_test):
    impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test = preprocess_test_user_data('../../data/MINDsmall_dev/behaviors.tsv')
    
    user_browsed_news_test = np.zeros((len(user_browsed_test), Config.max_browsed, Config.num_filters), dtype='float32')
    for i, user_browsed in enumerate(user_browsed_test):
        j = 0
        for news in user_browsed:
            if j < Config.max_browsed:
                user_browsed_news_test[i][j] = news_r_test[news_index_test[news]]
            j += 1
    all_candidate_news_test = np.array([news_r_test[news_index_test[i[0]]] for i in all_candidate_test])
    all_label_test = np.array(all_label_test)
    return user_browsed_news_test, all_candidate_news_test, all_label_test, impression_index, user_index, all_user_test


def read_json(file='embedding_matrix.json'):
    with open(file, 'r') as f:
        embedding_matrix = json.load(f)
    return embedding_matrix


def get_embedding_matrix(word_index):
    if os.path.exists('embedding_matrix.json'):
        print('Load embedding matrix...')
        embedding_matrix = np.array(read_json())
    return embedding_matrix


def train(model, train_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    for epoch in range(Config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.epochs))
        loss_epoch = []
        model.train()
        for i, data in enumerate(train_iter):
            browsed, candidate, labels= data
            optimizer.zero_grad()
            output = model(browsed, candidate)
            loss = torch.stack([x[0] for x in - F.log_softmax(output, dim=1)]).mean()
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Average Loss: {2:>5.2}'
                print(msg.format(i+1, loss.item(), np.mean(loss_epoch)))
        # print('saving model to model_%s.pkl...' % format(epoch))
        # torch.save(model.state_dict(), 'model_%s.pkl' % format(epoch))


if __name__ == "__main__":
    word_index = np.load('news/word_index.npy', allow_pickle=True).item()
    category_map = np.load('news/category_map.npy', allow_pickle=True).item()
    subcategory_map = np.load('news/subcategory_map.npy', allow_pickle=True).item()
    news_category = np.load('news/news_category.npy', allow_pickle=True)
    news_subcategory = np.load('news/news_subcategory.npy', allow_pickle=True)
    news_abstract = np.load('news/news_abstract.npy', allow_pickle=True)
    news_title = np.load('news/news_title.npy', allow_pickle=True)
    news_index = np.load('news/news_index.npy', allow_pickle=True).item()
    news_index_test = np.load('news/news_index_test.npy', allow_pickle=True).item()
    all_news_test = np.load('news/all_news_test.npy', allow_pickle=True)

    all_browsed, all_candidate, all_label = get_train_input(news_category, news_subcategory, \
                                                    news_abstract, news_title, news_index)
    pretrained_embedding = torch.from_numpy(get_embedding_matrix(word_index)).float()

    dataset = MyDataset(all_browsed, all_candidate, all_label)
    train_data = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    model = NAML(Config, pretrained_embedding).to(device)
    # print(model)
    train(model, train_data)

