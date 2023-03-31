import pickle
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from fine_row_cv_userprofile_classify import getFeaturesEncode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


class SearchDataset(Dataset):
    "获取数据的loader"
    def __init__(self, pkl_file, tokenizer, max_length=64, pkl_examples_limit=-1) -> None:
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f, encoding='bytes')
            if pkl_examples_limit !=-1:
                self.data = self.data[:pkl_examples_limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.featuresEncode = getFeaturesEncode()

        self.hash_topic_id = self.featuresEncode["topic_id"]["hash_info"]
        self.hash_topic_cate = self.featuresEncode["topic_cate"]["hash_info"]
        self.hash_cate = self.featuresEncode["cate"]["hash_info"]
        self.hash_subcate = self.featuresEncode["subcate"]["hash_info"]
        self.hash_user_city = self.featuresEncode["user_city"]["hash_info"]
        self.hash_gender = self.featuresEncode["gender"]["hash_info"]
        self.hash_decoration_status = self.featuresEncode["decoration_status"]["hash_info"]
        self.hash_wiki = self.featuresEncode["wiki"]["hash_info"]
        self.hash_user_identity_type = self.featuresEncode["user_identity_type"]["hash_info"]
        self.hash_obj_type = self.featuresEncode["obj_type"]["hash_info"]
        self.hash_uid = self.featuresEncode["uid"]["hash_info"]
        self.hash_obj_id = self.featuresEncode["obj_id"]["hash_info"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, set_type="train"):
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
        label = torch.squeeze(torch.tensor(int(self.data[index][1]), dtype=torch.long))

        doc_id = int(self.data[index][0])
        title_tf_num = float(self.data[index][5])
        remark_tf_num = float(self.data[index][6])

        favorite_num = float(self.data[index][7])
        like_num = float(self.data[index][8])
        comment_num = float(self.data[index][9])
        score = float(self.data[index][10])
        interval_days = float(self.data[index][11])
        wiki = str(self.data[index][12])
        user_identity_type = str(self.data[index][13])
        obj_type = str(self.data[index][14])

        if set_type == "dev":
            position = str(0)
        else:
            position = str(self.data[index][15])

        gender = str(self.data[index][16])
        age = float(self.data[index][17])
        uid_for_rerank = float(self.data[index][18])

        topic_id = str(self.data[index][19])
        topic_cate = str(self.data[index][20])
        cate = str(self.data[index][21])
        subcate = str(self.data[index][22])
        author_uid_for_rerank = str(self.data[index][23])

        author_type = str(self.data[index][24])
        author_city = str(self.data[index][25])
        author_gender = str(self.data[index][26])
        user_city = str(self.data[index][27])
        decoration_status = str(self.data[index][28])

        item_like_7day = float(self.data[index][29])
        item_comment_7day = float(self.data[index][30])
        item_favorite_7day = float(self.data[index][31])
        item_click_7day = float(self.data[index][32])
        item_like_30day = float(self.data[index][33])
        item_comment_30day = float(self.data[index][34])
        item_favorite_30day = float(self.data[index][35])
        item_click_30day = float(self.data[index][36])
        item_exp_30day = float(self.data[index][37])
        author_becomment = float(self.data[index][38])
        author_befollow = float(self.data[index][39])
        author_belike = float(self.data[index][40])
        author_befavorite = float(self.data[index][41])

        action_count_like = float(self.data[index][42])
        action_count_comment = float(self.data[index][43])
        action_count_favor = float(self.data[index][44])
        action_count_all_active = float(self.data[index][45])
        item_exp_7day = float(self.data[index][46])

        valid_click_doc_id_seq = self.data[index][47]
        valid_click_topic_id_seq = self.data[index][55]
        valid_click_topic_cate_seq = self.data[index][56]
        valid_click_cate_seq = self.data[index][57]
        valid_click_subcate_seq = self.data[index][58]
        valid_click_author_city_seq = self.data[index][60]
        valid_click_author_uid_seq = self.data[index][62]
        search_word_history_seq = self.data[index][77]

        isDeleteValidClickSeq = self.data[index][78]
        isDeleteSearchSeq = self.data[index][79]

        queryIndex = self.data[index][80]

        # 分类标签转化成正整数
        topic_integer = 0
        if topic_id in self.hash_topic_id:
            topic_integer = self.featuresEncode["topic_id"]["hash_info"][topic_id]

        topic_cate_integer = 0
        if topic_cate in self.hash_topic_cate:
            topic_cate_integer = self.hash_topic_cate[topic_cate]

        cate_integer = 0
        if cate in self.hash_cate:
            cate_integer = self.hash_cate[cate]

        subcate_integer = 0
        if subcate in self.hash_subcate:
            subcate_integer = self.hash_subcate[subcate]

        user_city_integer = 0
        if user_city in self.hash_user_city:
            user_city_integer = self.hash_user_city[user_city]

        auth_city_integer = 0
        if author_city in self.hash_user_city:
            auth_city_integer = self.hash_user_city[author_city]

        gender_integer = 0
        if gender in self.hash_gender:
            gender_integer = self.hash_gender[gender]

        author_gender_integer = 0
        if author_gender in self.hash_gender:
            author_gender_integer = self.hash_gender[author_gender]

        decoration_status_integer = 0
        if decoration_status in self.hash_decoration_status:
            decoration_status_integer = self.hash_decoration_status[decoration_status]

        wiki_integer = 0
        if wiki in self.hash_wiki:
            wiki_integer = self.hash_wiki[wiki]

        obj_type_integer = 0
        if obj_type in self.hash_obj_type:
            obj_type_integer = self.hash_obj_type[obj_type]

        user_identity_type_integer = 0
        if user_identity_type in self.hash_user_identity_type:
            user_identity_type_integer = self.hash_user_identity_type[user_identity_type]

        author_type_integer = 0
        if author_type in self.hash_user_identity_type:
            author_type_integer = self.hash_user_identity_type[author_type]

        '''if index < 1 :
            print("======query=======")
            print(query)
            print("======query=======")

            print("======title=======")
            print(doc_title)
            print("======title=======")

            print("======remark=======")
            print(doc_remark)
            print("======remark=======")

            print("======doc_id=======")
            print(doc_id)
            print("======doc_id=======")

            print("======label=======")
            print(label)
            print("======label=======")

            print("======title_tf_num=======")
            print(title_tf_num)
            print("======title_tf_num=======")

            print("======remark_tf_num=======")
            print(remark_tf_num)
            print("======remark_tf_num=======")

            print("======favorite_num=======")
            print(favorite_num)
            print("======favorite_num=======")

            print("======like_num=======")
            print(like_num)
            print("======like_num=======")

            print("======comment_num=======")
            print(comment_num)
            print("======comment_num=======")

            print("======score=======")
            print(score)
            print("======score=======")

            print("======interval_days=======")
            print(interval_days)
            print("======interval_days=======")

            print("======wiki_integer=======")
            print(wiki_integer)
            print("======wiki_integer=======")

            print("======user_identity_type_integer=======")
            print(user_identity_type_integer)
            print("======user_identity_type_integer=======")

            print("======obj_type_integer=======")
            print(obj_type_integer)
            print("======obj_type_integer=======")

            print("======position=======")
            print(position)
            print("======position=======")

            print("======gender_integer=======")
            print(gender_integer)
            print("======gender_integer=======")

            print("======age=======")
            print(age)
            print("======age=======")

            print("======uid_for_rerank=======")
            print(uid_for_rerank)
            print("======uid_for_rerank=======")

            print("======topic_integer=======")
            print(topic_integer)
            print("======topic_integer=======")

            print("======topic_cate_integer=======")
            print(topic_cate_integer)
            print("======topic_cate_integer=======")

            print("======cate_integer=======")
            print(cate_integer)
            print("======cate_integer=======")

            print("======subcate_integer=======")
            print(subcate_integer)
            print("======subcate_integer=======")

            print("======author_uid_for_rerank=======")
            print(author_uid_for_rerank)
            print("======author_uid_for_rerank=======")

            print("======author_type_integer=======")
            print(author_type_integer)
            print("======author_type_integer=======")

            print("======auth_city_integer=======")
            print(auth_city_integer)
            print("======auth_city_integer=======")

            print("======author_gender_integer=======")
            print(author_gender_integer)
            print("======author_gender_integer=======")

            print("======user_city_integer=======")
            print(user_city_integer)
            print("======user_city_integer=======")

            print("======decoration_status_integer=======")
            print(decoration_status_integer)
            print("======decoration_status_integer=======")

            print("======item_like_7day=======")
            print(item_like_7day)
            print("======item_like_7day=======")

            print("======item_comment_7day=======")
            print(item_comment_7day)
            print("======item_comment_7day=======")

            print("======item_favorite_7day=======")
            print(item_favorite_7day)
            print("======item_favorite_7day=======")

            print("======item_click_7day=======")
            print(item_click_7day)
            print("======item_click_7day=======")

            print("======item_like_30day=======")
            print(item_like_30day)
            print("======item_like_30day=======")

            print("======item_comment_30day=======")
            print(item_comment_30day)
            print("======item_comment_30day=======")

            print("======item_favorite_30day=======")
            print(item_favorite_30day)
            print("======item_favorite_30day=======")

            print("======item_click_30day=======")
            print(item_click_30day)
            print("======item_click_30day=======")

            print("======item_exp_30day=======")
            print(item_exp_30day)
            print("======item_exp_30day=======")

            print("======author_becomment=======")
            print(author_becomment)
            print("======author_becomment=======")

            print("======author_befollow=======")
            print(author_befollow)
            print("======author_befollow=======")

            print("======author_belike=======")
            print(author_belike)
            print("======author_belike=======")

            print("======author_befavorite=======")
            print(author_befavorite)
            print("======author_befavorite=======")
            print("======action_count_like=======")
            print(action_count_like)
            print("======action_count_like=======")

            print("======action_count_comment=======")
            print(action_count_comment)
            print("======action_count_comment=======")

            print("======action_count_favor=======")
            print(action_count_favor)
            print("======action_count_favor=======")
            print("======action_count_all_active=======")
            print(action_count_all_active)
            print("======action_count_all_active=======")

            print("======item_exp_7day=======")
            print(item_exp_7day)
            print("======item_exp_7day=======")

            print("======queryIndex=======")
            print(queryIndex)
            print("======queryIndex=======")

            print("======valid_click_doc_id_seq=======")
            print(valid_click_doc_id_seq)
            print("======valid_click_doc_id_seq=======")

            print("======valid_click_topic_id_seq=======")
            print(valid_click_topic_id_seq)
            print("======valid_click_topic_id_seq=======")

            print("======valid_click_topic_cate_seq=======")
            print(valid_click_topic_cate_seq)
            print("======valid_click_topic_cate_seq=======")

            print("======valid_click_cate_seq=======")
            print(valid_click_cate_seq)
            print("======valid_click_cate_seq=======")

            print("======valid_click_subcate_seq=======")
            print(valid_click_subcate_seq)
            print("======valid_click_subcate_seq=======")

            print("======valid_click_author_city_seq=======")
            print(valid_click_author_city_seq)
            print("======valid_click_author_city_seq=======")

            print("======valid_click_author_uid_seq=======")
            print(valid_click_author_uid_seq)
            print("======valid_click_author_uid_seq=======")

            print("======search_word_history_seq=======")
            print(search_word_history_seq)
            print("======search_word_history_seq=======")'''        

        encoder_dict = OrderedDict()
        encoder_dict["input_ids"] = input_ids.to(device)
        encoder_dict["attention_mask"] = attention_mask.to(device)
        encoder_dict["token_type_ids"] = token_type_ids.to(device)
        encoder_dict["label"] = label.to(device)

        
        # return input_ids, attention_mask, token_type_ids, label
        return encoder_dict