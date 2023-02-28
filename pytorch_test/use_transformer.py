from transformers import AutoModel, AutoTokenizer,AutoConfig


my_config = AutoConfig.from_pretrained("distilbert-base-uncased",n_head=3)

my_model = AutoModel.from_config(my_config)

my_tokenzier = AutoTokenizer.from_pretrained("distilbert-base-uncased")





