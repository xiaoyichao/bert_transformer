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

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        
        encoded_dict = self.tokenizer(self.data[index][0])
        input_ids = torch.squeeze(torch.tensor(encoded_dict['input_ids'],  dtype=torch.float32))
        attention_mask = torch.squeeze(torch.tensor(encoded_dict["attention_mask"],  dtype=torch.float32))
        token_type_ids = torch.squeeze(torch.tensor(encoded_dict["token_type_ids"],  dtype=torch.float32))
        label = torch.squeeze(torch.tensor(self.data[index][1], dtype=torch.float32))


        encoder_dict = OrderedDict()

        encoder_dict["input_ids"] = input_ids.to(device)
        encoder_dict["attention_mask"] = attention_mask.to(device)
        encoder_dict["token_type_ids"] = token_type_ids.to(device)
        encoder_dict["label"] = label.to(device)

        return encoder_dict
        

