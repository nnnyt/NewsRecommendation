class Config():
    model_name = 'DKN'
    max_title_len = 30
    embedding_dim = 300
    entity_embedding_dim = 100
    max_browsed = 50
    neg_sample = 1
    word_num = 41765 + 1
    nb_head = 15
    attention_dim = 200
    dropout = 0.2
    learning_rate = 0.0001
    epochs = 10
    batch_size = 64
    num_workers = 4
    embedding_num = 33795 + 1
    num_filters = 100
    window_size = [2, 3, 4]