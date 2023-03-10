from transformers import BertTokenizer
import torch


class ThreePieceTokenizer(BertTokenizer):
    def three_piece_encode_plus(
        self, text1: str, text2: str, text3: str, 
        max_length: int = None,
        add_special_tokens: bool = True,
        pad_to_max_length: bool = False,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
        return_tensors: str = None,
        **kwargs
    ):
        # Tokenize the three pieces of text separately
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        tokens3 = self.tokenize(text3)

        # Account for [CLS], [SEP], [SEP] , [SEP] and their special tokens
        special_tokens_count = 4 if add_special_tokens else 0
        if max_length is not None and max_length < len(tokens1) + len(tokens2) + len(tokens3) + special_tokens_count:
            # Need to truncate tokens
            # 这个位置要修改
            diff = len(tokens1) + len(tokens2) + len(tokens3) + special_tokens_count - max_length
            if diff > 0:
                tokens3 = tokens3[:-diff] if len(tokens3) > diff else []

                diff = len(tokens1) + len(tokens2) + len(tokens3) + special_tokens_count - max_length - 1
                if diff > 0:
                    tokens2 = tokens2[:-diff] if len(tokens2) > diff else []

                    diff = len(tokens1) + len(tokens2) + len(tokens3) + special_tokens_count - max_length - 2
                    if diff > 0:
                        tokens1 = tokens1[:-diff] if len(tokens1) > diff else []
            


        # Add special tokens if needed
        if add_special_tokens:
            if tokens1 != []:
                tokens = [self.cls_token] + tokens1 + [self.sep_token]
                token_type_ids = [0] * (len(tokens1) + 2)
            if tokens2 != []:
                tokens = tokens + tokens2 + [self.sep_token]
                token_type_ids = token_type_ids + [1] * (len(tokens2) + 1)
            if tokens3 != []:
                tokens = tokens + tokens3 + [self.sep_token]
                token_type_ids = token_type_ids + [2] * (len(tokens3) + 1)
        else:
            token_type_ids = [0] * len(tokens)

        # Convert tokens to ids
        input_ids = self.convert_tokens_to_ids(tokens)

        # Pad or truncate if needed
        if pad_to_max_length:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids)
            token_type_ids = token_type_ids + [self.pad_token_id] * padding_length
        else:
            attention_mask = [1] * len(input_ids)

        # Prepare output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_token_type_ids:
            output["token_type_ids"] = token_type_ids

        # Prepare tensors if needed
        if return_tensors is not None:
            output = {k: torch.tensor(v).unsqueeze(0) for k, v in output.items()}

        return output


if __name__ == "__main__":
    my_bert_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"

    tokenizer = ThreePieceTokenizer.from_pretrained(my_bert_path)
    query = "问题"*6
    doc_title = "标题"
    doc_remark = "正文"*62
    encoded_dict = tokenizer.three_piece_encode_plus(query,
                                                        doc_title, doc_remark,
                                                        truncation=True,
                                                        max_length=64,
                                                        add_special_tokens=True,
                                                        pad_to_max_length=True,
                                                        return_attention_mask=True,
                                                        return_tensors="pt")
