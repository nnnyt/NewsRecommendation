import numpy as np
import json
import os
import random
from config import config


def preprocess_user_data(filename):
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, "r") as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 1)
    use_data = data[:use_num]
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        browsed_news.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        impression_news.append(impressions)
    impression_pos = []
    impression_neg = []
    for impressions in impression_news:
        pos = []
        neg = []
        for news in impressions:
            if int(news[1]) == 1:
                pos.append(news[0])
            else:
                neg.append(news[0])
        impression_pos.append(pos)
        impression_neg.append(neg)
    all_browsed_news = []
    all_click = []
    all_unclick = []
    all_candidate = []
    all_label = []
    for k in range(len(browsed_news)):
        browsed = browsed_news[k]
        pos = impression_pos[k]
        neg = impression_neg[k]
        for pos_news in pos:
            all_browsed_news.append(browsed)
            all_click.append([pos_news])
            neg_news = random.sample(neg, config.neg_sample)
            all_unclick.append(neg_news)
            all_candidate.append([pos_news]+neg_news)
            all_label.append([1] + [0] * config.neg_sample)
            
    print('original behavior: ', len(browsed_news))
    print('processed behavior: ', len(all_browsed_news))
    return all_browsed_news, all_click, all_unclick, all_candidate, all_label


def preprocess_test_user_data(filename):
    print("Preprocessing test user data...")
    with open(filename, 'r') as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.1)
    use_data = data[:use_num]
    impression_index = []
    user_browsed_test = []
    all_candidate_test = []
    all_label_test = []
    user_index = {}
    all_user_test = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        if userID not in user_index:
            user_index[userID] = len(user_index)
            history = history.split()
            user_browsed_test.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        begin = len(all_candidate_test)
        end = len(impressions) + begin
        impression_index.append([begin, end])
        for news in impressions:
            all_user_test.append(userID)
            all_candidate_test.append([news[0]])
            all_label_test.append([int(news[1])])
    print('test samples: ', len(all_label_test))
    print('Found %s unique users.' % len(user_index))
    return impression_index, user_index, user_browsed_test, all_user_test, all_candidate_test, all_label_test