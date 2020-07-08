class Config():
    model_name = 'LSTUR'
    max_title_len = 30
    max_abstract_len = 100
    embedding_dim = 300
    category_embedding_dim = 100
    max_browsed = 50
    neg_sample = 1
    word_num = 72311 + 1
    category_num = 18 + 1
    subcategory_num = 270 + 1
    nb_head = 15
    attention_dim = 200
    dropout = 0.2
    learning_rate = 0.0001
    epochs = 10
    batch_size = 64
    num_workers = 8
    num_filters = 400
    window_size = 3
    model_type = 'ini'
    user_num = 56161 + 1
    masking_probability = 0.5