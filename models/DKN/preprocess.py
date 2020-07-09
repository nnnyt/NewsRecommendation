import numpy as np
import json
import os
import random
from config import Config


def preprocess_user_data(filename):
    print("Preprocessing user data...")
    browsed_news = []
    impression_news = []
    with open(filename, "r") as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.0001)
    use_data = data[:use_num]
    all_browsed_news = []
    all_candidate = []
    all_label = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        browsed_news.append(history)
        impressions = [x.split('-') for x in impressions.split()]
        pos = []
        neg = []
        for news in impressions:
            if int(news[1]) == 1:
                pos.append(news[0])
            else:
                neg.append(news[0])
        num = len(pos)
        try:
            neg = random.sample(neg, num)
        except:
            neg = neg
        for news in pos:
            all_browsed_news.append(history)
            all_candidate.append([news])
            all_label.append([1])
        for news in neg:
            all_browsed_news.append(history)
            all_candidate.append([news])
            all_label.append([0])

    random.seed(212)
    random.shuffle(all_browsed_news)
    random.seed(212)
    random.shuffle(all_candidate)
    random.seed(212)
    random.shuffle(all_label)            
    print('original behavior: ', len(browsed_news))
    print('processed behavior: ', len(all_browsed_news))
    return all_browsed_news, all_candidate, all_label


def preprocess_test_user_data(filename):
    print("Preprocessing test user data...")
    with open(filename, 'r') as f:
        data = f.readlines()
    random.seed(212)
    random.shuffle(data)
    use_num = int(len(data) * 0.0001)
    use_data = data[:use_num]
    impression_index = []
    all_browsed_test = []
    all_candidate_test = []
    all_label_test = []
    for l in use_data:
        userID, time, history, impressions = l.strip('\n').split('\t')
        history = history.split()
        impressions = [x.split('-') for x in impressions.split()]
        begin = len(all_candidate_test)
        end = len(impressions) + begin
        impression_index.append([begin, end])
        for news in impressions:
            all_browsed_test.append(history)
            all_candidate_test.append([news[0]])
            all_label_test.append([int(news[1])])
    print('test samples: ', len(all_label_test))
    return impression_index,all_browsed_test, all_candidate_test, all_label_test