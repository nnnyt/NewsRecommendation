class Config():
    model_name = 'TANR'
    max_title_len = 30
    embedding_dim = 300
    max_browsed = 50
    neg_sample = 1
    word_num = 41765 + 1
    category_num = 18
    attention_dim = 200
    dropout = 0.2
    learning_rate = 0.0001
    epochs = 10
    batch_size = 64
    num_workers = 8
    num_filters = 400
    window_size = 3
    category_loss_weight = 0.2
