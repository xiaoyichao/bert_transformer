
from keybert import KeyBERT

model_path = "./models/distilbert_torch"
model_bert = KeyBERT(model_path)

query_with_space = "美式 地砖 搭配"
list_with_tuple = model_bert.extract_keywords(
    query_with_space, keyphrase_ngram_range=(1, 1), top_n=10,)

term_weight_dict = {}
term_list = []

for pos, tuple_data in enumerate(list_with_tuple):
    term = tuple_data[0]
    ori_weight = tuple_data[1]
    term_list.append(term)
    if ori_weight >= 0.5 or pos == 0:
        term_weight = 2
        term_weight_dict[term] = term_weight
    else:
        term_weight = 1
        term_weight_dict[term] = term_weight
print( term_weight_dict)
# for term in split_words_list:  # 默认的忽略词和停止词修改为限定词
#     if term not in term_list:
#         term_weight = self.limit_word
#         term_weight_dict[term] = term_weight
