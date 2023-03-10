import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SearchDataset(Dataset):
    "获取数据的loader"
    def __init__(self, pkl_file, tokenizer, max_length=64, pkl_examples_limit=-1) -> None:
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f, encoding='bytes')
            if pkl_examples_limit !=-1:
                self.data = self.data[:pkl_examples_limit]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 在这个位置加入处理数据的逻辑
        query = self.data[index][2]
        doc_title = self.data[index][3]
        doc_remark = self.data[index][4]

        encoded_dict = self.tokenizer.three_piece_encode_plus(query,
                                                         doc_title, doc_remark,
                                                         truncation=True,
                                                         max_length=self.max_length,
                                                         add_special_tokens=True,
                                                         pad_to_max_length=True,
                                                         return_attention_mask=True,
                                                         return_tensors="pt")

        input_ids = torch.squeeze(encoded_dict['input_ids'])
        attention_mask = torch.squeeze(encoded_dict['attention_mask'])
        token_type_ids = torch.squeeze(encoded_dict['token_type_ids'])
        label = torch.squeeze(torch.tensor(int(self.data[index][1]), dtype=torch.int64))
        
        return input_ids, attention_mask, token_type_ids, label