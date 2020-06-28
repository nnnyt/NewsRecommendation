class Config():
    model_name = 'NRMS'
    max_title_len = 30
    max_abstract_len = 100
    embedding_dim = 300
    category_embedding_dim = 100
    max_browsed = 50
    neg_sample = 1
    word_num = 41765 + 1
    category_num = 17 + 1
    subcategory_num = 264 + 1
    nb_head = 15
    attention_dim = 200
    dropout = 0.2
    learning_rate = 0.0001
    epochs = 5
    batch_size = 64
    num_workers = 8
    num_filters = 400
    window_size = 3
    
    def set_category_num(self, category_map):
        self.category_num = len(category_map) + 1
        print('category num: ', self.category_num)
    
    def set_subcategory_num(self, subcategory_map):
        self.subcategory_num = len(subcategory_map) + 1
        print('subcategory num: ', self.subcategory_num)
    
    def set_word_num(self, word_index):
        self.word_num = len(word_index) + 1
        print('word num: ', self.word_num)