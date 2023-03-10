"这个文件是想要转换模型的参数名，但是现在不想走这个路了，因此这个文件还有点问题，仅供参考"
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pickle
import common4bert
import copy

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig 
from transformers4token import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split



if torch.cuda.is_available():
    print("torch.cuda.current_device()", torch.cuda.current_device())
else:
    print("USE CPU")

data_dir_path = "/data/search_opt_model/topk_opt/rank_fine_row_cv_userprofile"
pkl_examples_limit = 120

# load tokenizer and model
my_distilbert_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"
distilbert_tokenizer = BertTokenizer.from_pretrained(my_distilbert_path)
distilbert_config = DistilBertConfig.from_pretrained(my_distilbert_path)
distilbert_model = DistilBertModel.from_pretrained(my_distilbert_path, config=distilbert_config)
print(list(distilbert_model.state_dict().keys()))

# distilbert_model_named_parameters = 

my_bert_path = "bert-base-chinese"
bert_tokenizer = BertTokenizer.from_pretrained(my_bert_path)
bert_config = BertConfig.from_pretrained(my_bert_path)
bert_config.num_hidden_layers= 2
bert_model = BertModel.from_pretrained(my_bert_path, config=bert_config)
# bert_model.load_state_dict(torch.load('bert_model.pth'))
print(list(bert_model.state_dict().keys()))

bert_parameter_names = list(bert_model.state_dict().keys())


bert_model_state_dict = copy.deepcopy(bert_model.state_dict())
map_dict = {
    'embeddings.word_embeddings.weight': 'embeddings.word_embeddings.weight',
    'embeddings.position_embeddings.weight': 'embeddings.position_embeddings.weight',
    'embeddings.token_type_embeddings.weight': 'embeddings.token_type_embeddings.weight',
    'embeddings.LayerNorm.weight': 'embeddings.LayerNorm.weight',
    'embeddings.LayerNorm.bias': 'embeddings.LayerNorm.bias',
    'transformer.layer.0.attention.q_lin.weight': 'encoder.layer.0.attention.self.query.weight',
    'transformer.layer.0.attention.q_lin.bias': 'encoder.layer.0.attention.self.query.bias',
    'transformer.layer.0.attention.k_lin.weight': 'encoder.layer.0.attention.self.key.weight',
    'transformer.layer.0.attention.k_lin.bias': 'encoder.layer.0.attention.self.key.bias',
    'transformer.layer.0.attention.v_lin.weight': 'encoder.layer.0.attention.self.value.weight',
    'transformer.layer.0.attention.v_lin.bias': 'encoder.layer.0.attention.self.value.bias',
    'transformer.layer.0.attention.out_lin.weight': 'encoder.layer.0.attention.output.dense.weight',
    'transformer.layer.0.attention.out_lin.bias': 'encoder.layer.0.attention.output.dense.bias',
    'transformer.layer.0.sa_layer_norm.weight': 'encoder.layer.0.attention.output.LayerNorm.weight',
    'transformer.layer.0.sa_layer_norm.bias': 'encoder.layer.0.attention.output.LayerNorm.bias',
    'transformer.layer.0.ffn.lin1.weight': 'encoder.layer.0.intermediate.dense.weight',
    'transformer.layer.0.ffn.lin1.bias': 'encoder.layer.0.intermediate.dense.bias',
    'transformer.layer.0.ffn.lin2.weight': 'encoder.layer.0.output.dense.weight',
    'transformer.layer.0.ffn.lin2.bias': 'encoder.layer.0.output.dense.bias',
    'transformer.layer.0.output_layer_norm.weight': 'encoder.layer.0.output.LayerNorm.weight',
    'transformer.layer.0.output_layer_norm.bias': 'encoder.layer.0.output.LayerNorm.bias',

    'transformer.layer.1.attention.q_lin.weight': 'encoder.layer.1.attention.self.query.weight',
    'transformer.layer.1.attention.q_lin.bias': 'encoder.layer.1.attention.self.query.bias',
    'transformer.layer.1.attention.k_lin.weight': 'encoder.layer.1.attention.self.key.weight',
    'transformer.layer.1.attention.k_lin.bias': 'encoder.layer.1.attention.self.key.bias',
    'transformer.layer.1.attention.v_lin.weight': 'encoder.layer.1.attention.self.value.weight',
    'transformer.layer.1.attention.v_lin.bias': 'encoder.layer.1.attention.self.value.bias',
    'transformer.layer.1.attention.out_lin.weight': 'encoder.layer.1.attention.output.dense.weight',
    'transformer.layer.1.attention.out_lin.bias': 'encoder.layer.1.attention.output.dense.bias',
    'transformer.layer.1.sa_layer_norm.weight': 'encoder.layer.1.attention.output.LayerNorm.weight',
    'transformer.layer.1.sa_layer_norm.bias': 'encoder.layer.1.attention.output.LayerNorm.bias',
    'transformer.layer.1.ffn.lin1.weight': 'encoder.layer.1.intermediate.dense.weight',
    'transformer.layer.1.ffn.lin1.bias': 'encoder.layer.1.intermediate.dense.bias',
    'transformer.layer.1.ffn.lin2.weight': 'encoder.layer.1.output.dense.weight',
    'transformer.layer.1.ffn.lin2.bias': 'encoder.layer.1.output.dense.bias',
    'transformer.layer.1.output_layer_norm.weight': 'encoder.layer.1.output.LayerNorm.weight',
    'transformer.layer.1.output_layer_norm.bias': 'encoder.layer.1.output.LayerNorm.bias'


}
for name, param in distilbert_model.state_dict().items() :
    
    if name in map_dict:
        bert_name = map_dict[name]
        print(name, bert_name)
        value = param.detach().cpu().numpy()
        bert_model_state_dict[bert_name] = torch.FloatTensor(value)
    else:
        print("not in distilbert_model", name)
    


# import torch
# from transformers import BertModel, BertConfig

class MyBertModel(BertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.bert = BertModel(config)
        # Remove the pooler layer
        self.bert.pooler = None
        # self.bert_model.embeddings.token_type_embeddings 

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        return outputs

my_bert_config = BertConfig.from_pretrained(my_bert_path)


my_bert_config.num_hidden_layers= 2
my_bert_config.type_vocab_size = 8
my_bert_model = MyBertModel(my_bert_config)

my_bert_model.load_state_dict(bert_model_state_dict)

my_bert_model.save_pretrained("./models/bert-base-cased/")
bert_tokenizer.save_pretrained("./models/bert-base-cased/")
# torch.save(bert_model_state_dict, "./model.pt")
