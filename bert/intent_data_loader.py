import pickle
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


class IntentDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = 32

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        
        encoded_dict = self.tokenizer(self.data[index][0], padding = 'max_length', max_length = self.max_length, truncation=True, return_tensors='pt')
        input_ids = torch.squeeze(encoded_dict['input_ids'])
        attention_mask = torch.squeeze(encoded_dict["attention_mask"])
        token_type_ids = torch.squeeze(encoded_dict["token_type_ids"])
        label = torch.squeeze(torch.tensor(self.data[index][1], dtype=torch.int64))

        # input_ids = torch.squeeze(encoded_dict['input_ids']).half()
        # attention_mask = torch.squeeze(encoded_dict["attention_mask"]).half()
        # token_type_ids = torch.squeeze(encoded_dict["token_type_ids"]).half()
        # label = torch.squeeze(torch.tensor(self.data[index][1], dtype=torch.int64)).half()


        encoder_dict = OrderedDict()

        encoder_dict["input_ids"] = input_ids.to(device)
        encoder_dict["attention_mask"] = attention_mask.to(device)
        encoder_dict["token_type_ids"] = token_type_ids.to(device)
        encoder_dict["label"] = label.to(device)

        return encoder_dict
        


