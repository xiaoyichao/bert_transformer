# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:07:23
Description: 获取点击数据（正负样本比例1:4)并且处理，用于训练rank模型

conda activate py38
nohup python -u click_data4bert_by_day_1_4.py > click_data4bert_by_day_1_4.out 2>&1 &
tail -f click_data4bert_by_day_1_4.out
ps -ef | grep click_data4bert_by_day_1_4.py
'''

import pickle
import os

import sys
import configparser
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
os.chdir(sys.path[0])
sys.path.append("../")
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from presto_con.topresto import python_presto
import random
from common.common_log import Logger, get_log_name
from common.tool import Tool
from common.send_message import send_msg
from service.Tool.JiebaHHZ import JiebaHHZ
import numpy as np


current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

JiebaHHZ.loadUserDict()
JiebaHHZ.del_jieba_default_words()

def get_tf(query, sentence, have_jiebad=False, max_len=128):
    '''
    Author: xiaoyichao
    param {*}
    Description: 获取query 在doc中的词频
    '''
    tf_num = 0
    query_word_list = JiebaHHZ.SplitWords(query)
    if have_jiebad:
        sentence_word_list = sentence[:max_len]  # max_len个词
        sentence = "".join(sentence)
        sentence = sentence[:max_len]  # max_len个字
    else:
        sentence_word_list = JiebaHHZ.SplitWords(sentence[:max_len], isGetUnique = False)
    for query_word in query_word_list:
        for sentence_word in sentence_word_list:
            if sentence_word == query_word:
                tf_num += 1
    if query in sentence:  # 给 绿色-沙发 的 数据加分，防止“绿色的植物，沙发也很软”这样query的两个词分家的数据得到高分。
        tf_num += 1
    return tf_num


def standard_scaler(data):
    '''
    Author: xiaoyichao
    param {*}
    Description: 对数值型数据进行标准化，使其数据分数的均值为0附近 ，方差为1
    '''
    mm = StandardScaler()
    mm_data = mm.fit_transform(data)
    return mm_data


featureEncodeJsonFile = "/data/search_opt_model/topk_opt/fine_row_feature_encode_neg_v2.json"
fineRowHashDataJsonFile = "/data/search_opt_model/topk_opt/fine_row_hash_data_neg_v2.json"

class ClickData(object):
    '''
    Author: xiaoyichao
    param {*} self
    Description: 获取点击数据并且处理，用于训练rank模型
    '''

    def __init__(self):
        self.presto = python_presto()
        self.host = root_config["prosto"]['host']
        self.port = int(root_config["prosto"]['port'])
        self.user = root_config["prosto"]['user']
        self.password = root_config["prosto"]['password']

    def get_click_datas4bert(self, day):
        '''
        Author: xiaoyichao
        param {*}
        Description: 获取点击数据，目前需要增加对正负样本的比例控制
        直接用int 表示内容类型
            0 note
            1 整屋
            2 指南
            4 回答 note
            5 文章
        '''
        presto_con = self.presto.connect(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         password=self.password
                                         )
        log.logger.info("presto连接完成")
        dayTmp = day + 31
        timeday = day
        sql = '''with k as (select  
    query, 
    obj_id, 
    ln(1+max(favorite_num)) favorite_num, 
    ln(1+max(like_num)) like_num, 
    ln(1+max(comment_num)) comment_num, 
    max(score) score, 
    max(add_day) add_day, 
    max(wiki) wiki, 
    max(user_identity_type) user_identity_type,
    max(obj_type) obj_type, 
    min(position) position, 
    max(gender) gender,
    max(age) age, 
    max(is_valid) is_valid,
    max(relevance) relevance, 
    day, 
    uid, 
    max(search_request_id) search_request_id,
    max(topic_id) topic_id, 
    max(topic_cate) topic_cate, 
    max(cate) cate, 
    max(subcate) subcate, 
    max(author_uid) author_uid,
    max(author_type) author_type, 
    max(author_city) author_city, 
    max(author_gender) author_gender, 
    max(user_city) user_city, 
    max(decoration_status) decoration_status,
    ln(1+max(item_like_7day)) item_like_7day, 
    ln(1+max(item_comment_7day)) item_comment_7day,
  ln(1+max(item_favorite_7day)) item_favorite_7day, 
  ln(1+max(item_click_7day)) item_click_7day, 
  ln(1+max(item_exp_7day)) item_exp_7day, 
  ln(1+max(item_likek_30day)) item_likek_30day, 
  ln(1+max(item_comment_30day)) item_comment_30day, 
  ln(1+max(item_favorite_30day)) item_favorite_30day, 
  ln(1+max(item_click_30day)) item_click_30day,
  ln(1+max(item_exp_30day)) item_exp_30day, 
  ln(1+max(author_becomment)) author_becomment, 
  ln(1+max(author_befollow)) author_befollow, 
  ln(1+max(author_belike)) author_belike, 
  ln(1+max(author_befavorite)) author_befavorite, 
  ln(1+max(action_count_like)) action_count_like,
  ln(1+max(action_count_comment)) action_count_comment, 
  ln(1+max(action_count_favor)) action_count_favor, 
  ln(1+max(action_count_all_active)) action_count_all_active,
  'a' flag
  from oss.hhz.dws_search_user_personalization_rerank_daily_report  where
 day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not  null and obj_id is not null and relevance is not null and is_valid is not null group by query, obj_id, day, uid),
l as (
    select 
    min(favorite_num) min_favorite_num, 
     max(favorite_num) max_favorite_num, 
    min(like_num) min_like_num, 
     max(like_num)max_like_num, 
    min(comment_num) min_comment_num, 
     max(comment_num) max_comment_num, 
    min(item_like_7day) min_item_like_7day,
    max(item_like_7day) max_item_like_7day, 
    min(item_comment_7day) min_item_comment_7day,
     max(item_comment_7day) max_item_comment_7day,
    min(item_favorite_7day) min_item_favorite_7day, 
    max(item_favorite_7day) max_item_favorite_7day, 
    min(item_click_7day) min_item_click_7day, 
    max(item_click_7day) max_item_click_7day, 
    min(item_exp_7day) min_item_exp_7day, 
     max(item_exp_7day) max_item_exp_7day, 
    min(item_likek_30day) min_item_likek_30day, 
    max(item_likek_30day) max_item_likek_30day,
    min(item_comment_30day) min_item_comment_30day, 
    max(item_comment_30day) max_item_comment_30day, 
    min(item_favorite_30day) min_item_favorite_30day, 
    max(item_favorite_30day) max_item_favorite_30day, 
   min(item_click_30day) min_item_click_30day,
    max(item_click_30day) max_item_click_30day,
    min(item_exp_30day) min_item_exp_30day, 
    max(item_exp_30day) max_item_exp_30day, 
    min(author_becomment) min_author_becomment, 
    max(author_becomment) max_author_becomment, 
      min(author_befollow) min_author_befollow, 
      max(author_befollow) max_author_befollow, 
    min(author_belike) min_author_belike, 
    max(author_belike) max_author_belike, 
      min(author_befavorite) min_author_befavorite, 
      max(author_befavorite) max_author_befavorite, 
       min(action_count_like) min_action_count_like,
       max(action_count_like) max_action_count_like,
       min(action_count_comment) min_action_count_comment, 
       max(action_count_comment) max_action_count_comment, 
      min(action_count_favor) min_action_count_favor, 
      max(action_count_favor) max_action_count_favor, 
      min(action_count_all_active) min_action_count_all_active,
       max(action_count_all_active) max_action_count_all_active,
      'a' flag
    from k
)
select 
                        query as "搜索词",
                        obj_id as "内容id",
                        round((favorite_num - min_favorite_num)/ (max_favorite_num - min_favorite_num), 4) as "内容的总收藏数",
                        round((like_num - min_like_num)/ (max_like_num - min_like_num), 4) as "内容的总点赞数",
                        round((comment_num - min_comment_num)/ (max_comment_num - min_comment_num), 4) as "内容的总评论数",
                        score as "内容的后台质量分数",
                        add_day as "内容的首次发布时间",
                        wiki as "是否挂了wiki",
                        user_identity_type as "发布内容的用户身份",
                        obj_type as "内容类型",
                        position as "内容曝光位置",
                        gender as "性别",
                        age as "年龄",
                        is_valid as "是否 是有效点击",
                        relevance as "相关性得分",
                        day as "搜索行为发生的日期",
                        uid as "用户uid",
                        search_request_id as "search id",
                        topic_id as "话题id",
                        topic_cate as "话题分类",
                        cate as "内容一级分类",
                        subcate as "内容二级分类",
                        author_uid as "作者uid",
                        author_type as "作者类型",
                        author_city as "作者所在城市",
                        author_gender as "作者性别",
                        user_city as "用户所住城市",
                        decoration_status as "装修状态",
                        round((item_like_7day - min_item_like_7day)/ (max_item_like_7day- min_item_like_7day), 4) as "内容近7天被点赞数",
                        round((item_comment_7day - min_item_comment_7day)/ (max_item_comment_7day- min_item_comment_7day), 4) as "内容近7天被评论数",
                        round((item_favorite_7day - min_item_favorite_7day)/ (max_item_favorite_7day- min_item_favorite_7day), 4) as "内容近7天被收藏数",
                        round((item_click_7day - min_item_click_7day)/ (max_item_click_7day- min_item_click_7day), 4) as "内容近7天被点击数",
                        round((item_likek_30day - min_item_likek_30day)/ (max_item_likek_30day- min_item_likek_30day), 4) as "内容近30天被点赞数",
                        round((item_comment_30day - min_item_comment_30day)/ (max_item_comment_30day- min_item_comment_30day), 4) as "内容近30天被评论数",
                        round((item_favorite_30day - min_item_favorite_30day)/ (max_item_favorite_30day- min_item_favorite_30day), 4)  as "内容近30天被收藏数",
                        round((item_click_30day - min_item_click_30day)/ (max_item_click_30day- min_item_click_30day), 4)  as "内容近30天被点击数",
                        round((item_exp_30day - min_item_exp_30day)/ (max_item_exp_30day- min_item_exp_30day), 4) as "内容近30天被曝光数",
                        round((author_becomment - min_author_becomment)/ (max_author_becomment- min_author_becomment), 4)  as "作者内容被评论数",
                        round((author_befollow - min_author_befollow)/ (max_author_befollow- min_author_befollow), 4)  as "作者被关注数",
                        round((author_belike - min_author_belike)/ (max_author_belike- min_author_belike), 4)  as "作者内容被点赞数",
                        round((author_befavorite - min_author_befavorite)/ (max_author_befavorite- min_author_befavorite), 4)  as "作者内容被收藏数",
                        round((action_count_like - min_action_count_like)/ (max_action_count_like- min_action_count_like), 4)  as "用户内容被点赞数",
                        round((action_count_comment - min_action_count_comment)/ (max_action_count_comment- min_action_count_comment), 4)  as "用户内容被评论数",
                        round((action_count_favor - min_action_count_favor)/ (max_action_count_favor- min_action_count_favor), 4)  as "用户内容被收藏数",
                        round((action_count_all_active - min_action_count_all_active)/ (max_action_count_all_active- min_action_count_all_active), 4)  as "用户内容被交互数",
                        round((item_exp_7day - min_item_exp_7day)/ (max_item_exp_7day- min_item_exp_7day), 4)  as "内容近7天被曝光数"
 from k join l on k.flag=l.flag order by search_request_id, query, uid, position asc''' % (day, day)

        log.logger.info("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 完成")
        log.logger.info("开始解析数据")

        featureEncodeHashInfo = {
            "topic_id": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "topic_cate": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "cate": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info": {"空的空的": 0}
            },
            "subcate": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "user_city": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "gender": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "decoration_status": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "wiki": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "user_identity_type": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "obj_type": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info" : {"空的空的" : 0}
            },
            "search_word": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info": {"空的空的": 0}
            },
            "uid": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info": {"空的空的": 0}
            },
            "obj_id": {
                "val": ["空的空的"],
                "index": [0],
                "hash_info": {"空的空的": 0}
            }
        }

        if os.path.exists(featureEncodeJsonFile):
            with open(featureEncodeJsonFile, 'r') as load_f:
                featureEncodeHashInfo = json.load(load_f)

        fetchDatas = presto_cursor.fetchall()

        for info in fetchDatas:
            doc_id = info[1]
            doc_id = doc_id.strip()
            if len(doc_id) == 0:
                continue

            topicId = 0 if info[18] is None else info[18]
            topicId = "" if topicId == 0 else str(topicId)
            if len(topicId) > 0 and topicId not in featureEncodeHashInfo["topic_id"]["hash_info"]:
                featureEncodeHashInfo["topic_id"]["val"].append(topicId)
                index = len(featureEncodeHashInfo["topic_id"]["val"])

                featureEncodeHashInfo["topic_id"]["index"].append(index)
                featureEncodeHashInfo["topic_id"]["hash_info"][topicId] = index

            topicCate = info[19]
            topicCate = "" if topicCate is None else topicCate
            if len(topicCate) > 0 and topicCate not in featureEncodeHashInfo["topic_cate"]["hash_info"]:
                featureEncodeHashInfo["topic_cate"]["val"].append(topicCate)
                index = len(featureEncodeHashInfo["topic_cate"]["val"])

                featureEncodeHashInfo["topic_cate"]["index"].append(index)
                featureEncodeHashInfo["topic_cate"]["hash_info"][topicCate] = index

            cate = info[20]
            cate = "" if cate is None else cate
            if len(cate) > 0 and cate not in featureEncodeHashInfo["cate"]["hash_info"]:
                featureEncodeHashInfo["cate"]["val"].append(cate)
                index = len(featureEncodeHashInfo["cate"]["val"])

                featureEncodeHashInfo["cate"]["index"].append(index)
                featureEncodeHashInfo["cate"]["hash_info"][cate] = index

            subcate = info[21]
            subcate = "" if subcate is None else subcate
            if len(subcate) > 0 and subcate not in featureEncodeHashInfo["subcate"]["hash_info"]:
                featureEncodeHashInfo["subcate"]["val"].append(subcate)
                index = len(featureEncodeHashInfo["subcate"]["val"])

                featureEncodeHashInfo["subcate"]["index"].append(index)
                featureEncodeHashInfo["subcate"]["hash_info"][subcate] = index

            authorTypeStr = str(info[23])
            authorTypeStr = "" if authorTypeStr is None else authorTypeStr
            if len(authorTypeStr) > 0 and authorTypeStr not in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                featureEncodeHashInfo["user_identity_type"]["val"].append(authorTypeStr)
                index = len(featureEncodeHashInfo["user_identity_type"]["val"])

                featureEncodeHashInfo["user_identity_type"]["index"].append(index)
                featureEncodeHashInfo["user_identity_type"]["hash_info"][authorTypeStr] = index

            authorCityStr = str(info[24])
            authorCityStr = "" if authorCityStr is None else authorCityStr
            if len(authorCityStr) > 0 and authorCityStr not in featureEncodeHashInfo["user_city"]["hash_info"]:
                featureEncodeHashInfo["user_city"]["val"].append(authorCityStr)
                index = len(featureEncodeHashInfo["user_city"]["val"])

                featureEncodeHashInfo["user_city"]["index"].append(index)
                featureEncodeHashInfo["user_city"]["hash_info"][authorCityStr] = index

            authorGenderStr = str(info[25])
            authorGenderStr = "" if authorGenderStr is None else authorGenderStr
            if len(authorGenderStr) > 0 and authorGenderStr not in featureEncodeHashInfo["gender"]["hash_info"]:
                featureEncodeHashInfo["gender"]["val"].append(authorGenderStr)
                index = len(featureEncodeHashInfo["gender"]["val"])

                featureEncodeHashInfo["gender"]["index"].append(index)
                featureEncodeHashInfo["gender"]["hash_info"][authorGenderStr] = index

            userCityStr = str(info[26])
            userCityStr = "" if userCityStr is None else userCityStr
            if len(userCityStr) > 0 and userCityStr not in featureEncodeHashInfo["user_city"]["hash_info"]:
                featureEncodeHashInfo["user_city"]["val"].append(userCityStr)
                index = len(featureEncodeHashInfo["user_city"]["val"])

                featureEncodeHashInfo["user_city"]["index"].append(index)
                featureEncodeHashInfo["user_city"]["hash_info"][userCityStr] = index

            genderStr = str(info[11])
            genderStr = "" if genderStr is None else genderStr
            if len(genderStr) > 0 and genderStr not in featureEncodeHashInfo["gender"]["hash_info"]:
                featureEncodeHashInfo["gender"]["val"].append(genderStr)
                index = len(featureEncodeHashInfo["gender"]["val"])

                featureEncodeHashInfo["gender"]["index"].append(index)
                featureEncodeHashInfo["gender"]["hash_info"][genderStr] = index

            decorationStatusStr = str(info[27])
            decorationStatusStr = "" if decorationStatusStr is None else decorationStatusStr
            if len(decorationStatusStr) > 0 and decorationStatusStr not in featureEncodeHashInfo["decoration_status"]["hash_info"]:
                featureEncodeHashInfo["decoration_status"]["val"].append(decorationStatusStr)
                index = len(featureEncodeHashInfo["decoration_status"]["val"])

                featureEncodeHashInfo["decoration_status"]["index"].append(index)
                featureEncodeHashInfo["decoration_status"]["hash_info"][decorationStatusStr] = index

            wikiStr = str(info[7])
            wikiStr = "" if wikiStr is None else wikiStr
            if len(wikiStr) > 0 and wikiStr not in featureEncodeHashInfo["wiki"][
                "hash_info"]:
                featureEncodeHashInfo["wiki"]["val"].append(wikiStr)
                index = len(featureEncodeHashInfo["wiki"]["val"])

                featureEncodeHashInfo["wiki"]["index"].append(index)
                featureEncodeHashInfo["wiki"]["hash_info"][wikiStr] = index

            userIdentityTypeStr = "" if info[8] is None else str(info[8])
            if len(userIdentityTypeStr) > 0 and userIdentityTypeStr not in featureEncodeHashInfo["user_identity_type"][
                "hash_info"]:
                featureEncodeHashInfo["user_identity_type"]["val"].append(userIdentityTypeStr)
                index = len(featureEncodeHashInfo["user_identity_type"]["val"])

                featureEncodeHashInfo["user_identity_type"]["index"].append(index)
                featureEncodeHashInfo["user_identity_type"]["hash_info"][userIdentityTypeStr] = index

            objTypeStr = "" if info[9] is None else str(info[9])
            if len(objTypeStr) > 0 and objTypeStr not in featureEncodeHashInfo["obj_type"][
                "hash_info"]:
                featureEncodeHashInfo["obj_type"]["val"].append(objTypeStr)
                index = len(featureEncodeHashInfo["obj_type"]["val"])

                featureEncodeHashInfo["obj_type"]["index"].append(index)
                featureEncodeHashInfo["obj_type"]["hash_info"][objTypeStr] = index

        click_datas = {}

        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        obj_type_dict_keys = set(obj_type_dict.keys())

        i = 0

        for info in fetchDatas:
            query = info[0]

            doc_id = info[1]
            doc_id = doc_id.strip()
            if len(doc_id) == 0:
                continue

            favorite_num = info[2]
            like_num = info[3]
            comment_num = info[4]
            score = info[5]
            add_day = info[6]

            wiki = info[7]
            user_identity_type = info[8]
            objTypeStr = info[9]

            obj_type_num = Tool.getTypeByObjId(doc_id)

            position = info[10]
            gender = info[11]

            age = info[12]

            is_valid = info[13]
            # day = time.strftime("%Y-%m-%d", time.localtime()).replace("-", "")

            relevance = info[14]
            day = info[15]
            uid = info[16]
            search_request_id = info[17]

            topic_id = str(info[18])
            topic_cate = info[19]
            cate = info[20]
            subcate = info[21]

            author_uid = info[22]
            author_type = info[23]
            author_city = info[24]
            author_gender = info[25]

            user_city = info[26]
            decoration_status = info[27]

            item_like_7day = info[28]
            item_comment_7day = info[29]
            item_favorite_7day = info[30]
            item_click_7day = info[31]

            item_like_30day = info[32]
            item_comment_30day = info[33]
            item_favorite_30day = info[34]
            item_click_30day = info[35]
            item_exp_30day = info[36]

            author_becomment = info[37]
            author_befollow = info[38]
            author_belike = info[39]
            author_befavorite = info[40]

            action_count_like = info[41]
            action_count_comment = info[42]
            action_count_favor = info[43]
            action_count_all_active = info[44]

            item_exp_7day = info[45]

            '''数据预处理'''
            favorite_num = 0 if favorite_num is None else float(favorite_num)
            like_num = 0 if like_num is None else float(like_num)
            comment_num = 0 if comment_num is None else float(comment_num)

            obj_type_for_rerank = "空的空的" if objTypeStr is None else objTypeStr

            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict_keys else obj_type_dict[obj_type_num]
            obj_type = obj_type.strip()

            if obj_type == "note" or obj_type == "回答的note":
                if score is None:
                    score = 0
                else:
                    if score < 60:
                        score = 60

            score = 0 if score is None else score

            score_level = 0
            if score <= 10 :
                score_level = 1
            elif score <= 20:
                score_level = 2
            elif score <= 30:
                score_level = 3
            elif score <= 40:
                score_level = 4
            elif score <= 50:
                score_level = 5
            elif score <= 60:
                score_level = 6
            elif score <= 70:
                score_level = 7
            elif score <= 80:
                score_level = 8
            elif score <= 90:
                score_level = 9
            elif score > 90:
                score_level = 10

            if add_day and day:
                add_day = str(add_day)
                day = str(day)
                add_day = datetime(year=int(add_day[0:4]), month=int(
                    add_day[4:6]), day=int(add_day[6:8]))
                day = datetime(year=int(day[0:4]), month=int(
                    day[4:6]), day=int(day[6:8]))
                interval_days = (day - add_day).days
            else:
                interval_days = 0

            interval_week = interval_days // 7

            wiki_for_userprofile = "空的空的"
            if wiki  in featureEncodeHashInfo["wiki"]["hash_info"]:
                wiki_for_userprofile = wiki

            if user_identity_type not in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                user_identity_type = "空的空的"

            position = 0 if position is None else int(position)
            if position < 0:
                position = 0

            if gender not in featureEncodeHashInfo["gender"]["hash_info"]:
                gender = "空的空的"

            age = 0 if age is None else int(age)
            if age < 0:
                age = 0

            age_level = 0
            if age <= 10:
                age_level = 1
            elif age <= 20:
                age_level = 2
            elif age <= 30:
                age_level = 3
            elif age <= 40:
                age_level = 4
            elif age <= 50:
                age_level = 5
            elif age <= 60:
                age_level = 6
            elif age > 60:
                age_level = 7

            if topic_id not in featureEncodeHashInfo["topic_id"]["hash_info"]:
                topic_id = "空的空的"

            if topic_cate not in featureEncodeHashInfo["topic_cate"]["hash_info"]:
                topic_cate = "空的空的"

            if cate not in featureEncodeHashInfo["cate"]["hash_info"]:
                cate = "空的空的"

            if subcate not in featureEncodeHashInfo["subcate"]["hash_info"]:
                subcate = "空的空的"

            uid_for_rerank = str(uid)

            author_uid_for_rerank = str(author_uid)

            if author_type not in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                author_type = "空的空的"

            if author_city not in featureEncodeHashInfo["user_city"]["hash_info"]:
                author_city = "空的空的"

            if author_gender not in featureEncodeHashInfo["gender"]["hash_info"]:
                author_gender = "空的空的"

            if user_city not in featureEncodeHashInfo["user_city"]["hash_info"]:
                user_city = "空的空的"

            if decoration_status not in featureEncodeHashInfo["decoration_status"]["hash_info"]:
                decoration_status = "空的空的"

            item_like_7day = 0 if item_like_7day is None else float(item_like_7day)
            item_comment_7day = 0 if item_comment_7day is None else float(item_comment_7day)
            item_favorite_7day = 0 if item_favorite_7day is None else float(item_favorite_7day)
            item_click_7day = 0 if item_click_7day is None else float(item_click_7day)
            item_exp_7day = 0 if item_exp_7day is None else float(item_exp_7day)

            item_like_30day = 0 if item_like_30day is None else float(item_like_30day)
            item_comment_30day = 0 if item_comment_30day is None else float(item_comment_30day)
            item_favorite_30day = 0 if item_favorite_30day is None else float(item_favorite_30day)
            item_click_30day = 0 if item_click_30day is None else float(item_click_30day)
            item_exp_30day = 0 if item_exp_30day is None else float(item_exp_30day)

            author_becomment = 0 if author_becomment is None else float(author_becomment)
            author_befollow = 0 if author_befollow is None else float(author_befollow)
            author_belike = 0 if author_belike is None else float(author_belike)
            author_befavorite = 0 if author_befavorite is None else float(author_befavorite)

            action_count_like = 0 if action_count_like is None else float(action_count_like)
            action_count_comment = 0 if action_count_comment is None else float(action_count_comment)
            action_count_favor = 0 if action_count_favor is None else float(action_count_favor)
            action_count_all_active = 0 if action_count_all_active is None else float(action_count_all_active)

            query = str(query)

            if query not in click_datas:
                click_datas[query] = {
                    'query': query, 'uid' : uid,  'positive_list': [], 'negative_list': []}


            one_sample = [doc_id, relevance, favorite_num, like_num, comment_num, score_level, interval_week, wiki_for_userprofile,
             user_identity_type, obj_type_for_rerank, position, gender, age_level, uid_for_rerank, topic_id, topic_cate,
             cate, subcate, author_uid_for_rerank, author_type, author_city, author_gender, user_city, decoration_status, item_like_7day,
             item_comment_7day, item_favorite_7day, item_click_7day, item_like_30day, item_comment_30day, item_favorite_30day,
             item_click_30day, item_exp_30day, author_becomment, author_befollow, author_belike, author_befavorite, action_count_like,
             action_count_comment, action_count_favor, action_count_all_active, item_exp_7day]

            # 正样本
            if relevance > 0:
                if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                    click_datas[query]['positive_list'].append(one_sample)
                else:
                    # if relevance == 1: # 正负样本比例不对
                    #     relevance = is_valid  # 正负样本比例不对
                    # 正负样本比例对
                    if relevance == 1 and is_valid == 1:  # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                        click_datas[query]['positive_list'].append(one_sample)
                    else:
                        relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本
                        one_sample[1] = relevance
                        click_datas[query]['negative_list'].append(one_sample)
            # 负样本
            else:
                click_datas[query]['negative_list'].append(one_sample)

            if i % 10000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i += 1

        # print(click_datas)
        log.logger.info("完成解析数据")

        for query, click_data_format in  click_datas.items():
            for item in click_data_format["positive_list"]:
                corpus_obj_ids_set.add(item[0])

            for item in click_data_format["negative_list"]:
                corpus_obj_ids_set.add(item[0])

        presto_cursor.close()

        sql = '''
        with a as (select max(day) as max_day, obj_id from (select
                       obj_id,day 
                    from oss.hhz.dws_search_user_personalization_rerank_daily_report
                    where  day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not  null and obj_id is not null and relevance is not null and is_valid is not null  and (is_valid = 1 or relevance > 1)) group by obj_id),
b as (select
                      aa.obj_id,aa.favorite_num, aa.like_num,aa.comment_num,aa.score,aa.add_day, aa.wiki, aa.obj_type,aa.topic_id,aa.topic_cate,aa.cate,aa.subcate,aa.author_type,aa.author_city,aa.author_gender,aa.item_like_7day,aa.item_comment_7day,aa.item_favorite_7day,aa.item_click_7day,aa.item_exp_7day,aa.item_likek_30day,aa.item_comment_30day,
                      aa.item_favorite_30day,aa.item_click_30day,aa.item_exp_30day,aa.author_becomment,aa.author_befollow,aa.author_belike,aa.author_befavorite, aa.author_uid, aa.day
                    from a  left join oss.hhz.dws_search_user_personalization_rerank_daily_report aa on   a.max_day = aa.day and a.obj_id = aa.obj_id
                    where  aa.day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and aa.query is not  null and aa.obj_id is not null and aa.relevance is not null and aa.is_valid is not null  and (is_valid = 1 or relevance > 1))
select * from b
        ''' % (dayTmp + 1, timeday + 1, dayTmp + 1, timeday + 1)

        log.logger.info("presto_cursor 用户有效点击内容: " + sql)
        presto_valid_click_data_cursor = presto_con.cursor()
        log.logger.info("presto cursor 用户有效点击内容 完成")
        presto_valid_click_data_cursor.execute(sql)
        log.logger.info("presto 执行 用户有效点击内容 完成")
        log.logger.info("开始解析数据 用户有效点击内容")

        valid_click_datas = presto_valid_click_data_cursor.fetchall()
        for valid_click_data in valid_click_datas:
            doc_id = valid_click_data[0]
            doc_id = doc_id.strip()

            if len(doc_id) == 0:
                continue

            wiki = valid_click_data[6]
            wikiStr = str(wiki)
            wikiStr = "" if wikiStr is None else wikiStr
            if len(wikiStr) > 0 and wikiStr not in featureEncodeHashInfo["wiki"][
                "hash_info"]:
                featureEncodeHashInfo["wiki"]["val"].append(wikiStr)
                index = len(featureEncodeHashInfo["wiki"]["val"])

                featureEncodeHashInfo["wiki"]["index"].append(index)
                featureEncodeHashInfo["wiki"]["hash_info"][wikiStr] = index

            obj_type = valid_click_data[7]
            objTypeStr = "" if obj_type is None else str(obj_type)
            if len(objTypeStr) > 0 and objTypeStr not in featureEncodeHashInfo["obj_type"][
                "hash_info"]:
                featureEncodeHashInfo["obj_type"]["val"].append(objTypeStr)
                index = len(featureEncodeHashInfo["obj_type"]["val"])

                featureEncodeHashInfo["obj_type"]["index"].append(index)
                featureEncodeHashInfo["obj_type"]["hash_info"][objTypeStr] = index


            topicId = 0 if valid_click_data[8] is None else valid_click_data[8]
            topicId = "" if topicId == 0 else str(topicId)
            if len(topicId) > 0 and topicId not in featureEncodeHashInfo["topic_id"]["hash_info"]:
                featureEncodeHashInfo["topic_id"]["val"].append(topicId)
                index = len(featureEncodeHashInfo["topic_id"]["val"])

                featureEncodeHashInfo["topic_id"]["index"].append(index)
                featureEncodeHashInfo["topic_id"]["hash_info"][topicId] = index

            topic_cate = valid_click_data[9]
            topicCate = "" if topic_cate is None else topic_cate
            if len(topicCate) > 0 and topicCate not in featureEncodeHashInfo["topic_cate"]["hash_info"]:
                featureEncodeHashInfo["topic_cate"]["val"].append(topicCate)
                index = len(featureEncodeHashInfo["topic_cate"]["val"])

                featureEncodeHashInfo["topic_cate"]["index"].append(index)
                featureEncodeHashInfo["topic_cate"]["hash_info"][topicCate] = index

            cate = valid_click_data[10]
            cate = "" if cate is None else cate
            if len(cate) > 0 and cate not in featureEncodeHashInfo["cate"]["hash_info"]:
                featureEncodeHashInfo["cate"]["val"].append(cate)
                index = len(featureEncodeHashInfo["cate"]["val"])

                featureEncodeHashInfo["cate"]["index"].append(index)
                featureEncodeHashInfo["cate"]["hash_info"][cate] = index

            subcate = valid_click_data[11]
            subcate = "" if subcate is None else subcate
            if len(subcate) > 0 and subcate not in featureEncodeHashInfo["subcate"]["hash_info"]:
                featureEncodeHashInfo["subcate"]["val"].append(subcate)
                index = len(featureEncodeHashInfo["subcate"]["val"])

                featureEncodeHashInfo["subcate"]["index"].append(index)
                featureEncodeHashInfo["subcate"]["hash_info"][subcate] = index

            author_type = valid_click_data[12]
            authorTypeStr = str(author_type)
            authorTypeStr = "" if authorTypeStr is None else authorTypeStr
            if len(authorTypeStr) > 0 and authorTypeStr not in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                featureEncodeHashInfo["user_identity_type"]["val"].append(authorTypeStr)
                index = len(featureEncodeHashInfo["user_identity_type"]["val"])

                featureEncodeHashInfo["user_identity_type"]["index"].append(index)
                featureEncodeHashInfo["user_identity_type"]["hash_info"][authorTypeStr] = index

            author_city = valid_click_data[13]
            authorCityStr = str(author_city)
            authorCityStr = "" if authorCityStr is None else authorCityStr
            if len(authorCityStr) > 0 and authorCityStr not in featureEncodeHashInfo["user_city"]["hash_info"]:
                featureEncodeHashInfo["user_city"]["val"].append(authorCityStr)
                index = len(featureEncodeHashInfo["user_city"]["val"])

                featureEncodeHashInfo["user_city"]["index"].append(index)
                featureEncodeHashInfo["user_city"]["hash_info"][authorCityStr] = index


            author_gender = valid_click_data[14]
            authorGenderStr = str(author_gender)
            authorGenderStr = "" if authorGenderStr is None else authorGenderStr
            if len(authorGenderStr) > 0 and authorGenderStr not in featureEncodeHashInfo["gender"]["hash_info"]:
                featureEncodeHashInfo["gender"]["val"].append(authorGenderStr)
                index = len(featureEncodeHashInfo["gender"]["val"])

                featureEncodeHashInfo["gender"]["index"].append(index)
                featureEncodeHashInfo["gender"]["hash_info"][authorGenderStr] = index

        hashValidClickObjFeature = {}
        for valid_click_data in valid_click_datas:
            doc_id = valid_click_data[0]
            doc_id = doc_id.strip()

            if len(doc_id) == 0:
                continue

            favorite_num = valid_click_data[1]
            like_num = valid_click_data[2]
            comment_num = valid_click_data[3]
            score = valid_click_data[4]
            add_day = valid_click_data[5]
            wiki = valid_click_data[6]
            objTypeStr = valid_click_data[7]
            topic_id = str(valid_click_data[8])
            topic_cate = valid_click_data[9]

            cate = valid_click_data[10]

            subcate = valid_click_data[11]

            author_type = valid_click_data[12]

            author_city = valid_click_data[13]

            author_gender = valid_click_data[14]

            item_like_7day = valid_click_data[15]
            item_comment_7day = valid_click_data[16]
            item_favorite_7day = valid_click_data[17]
            item_click_7day = valid_click_data[18]
            item_exp_7day = valid_click_data[19]
            item_like_30day = valid_click_data[20]
            item_comment_30day = valid_click_data[21]
            item_favorite_30day = valid_click_data[22]
            item_click_30day = valid_click_data[23]
            item_exp_30day = valid_click_data[24]
            author_becomment = valid_click_data[25]
            author_befollow = valid_click_data[26]
            author_belike = valid_click_data[27]
            author_befavorite = valid_click_data[28]
            author_uid = valid_click_data[29]
            day = valid_click_data[30]

            '''数据预处理'''
            favorite_num = 0 if favorite_num is None else float(favorite_num)
            like_num = 0 if like_num is None else float(like_num)
            comment_num = 0 if comment_num is None else float(comment_num)

            obj_type_for_rerank = "空的空的" if objTypeStr is None else objTypeStr

            obj_type_num = Tool.getTypeByObjId(doc_id)
            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict_keys else obj_type_dict[obj_type_num]
            obj_type = obj_type.strip()
            if obj_type == "note" or obj_type == "回答的note":
                if score is None:
                    score = 0
                else:
                    if score < 60:
                        score = 60

            score = 0 if score is None else score

            score_level = 0
            if score > 1:
                score_level = 1
            elif score > 10:
                score_level = 2
            elif score > 20:
                score_level = 3
            elif score > 30:
                score_level = 4
            elif score > 40:
                score_level = 5
            elif score > 50:
                score_level = 6
            elif score > 60:
                score_level = 7
            elif score > 70:
                score_level = 8
            elif score > 80:
                score_level = 9
            elif score > 90:
                score_level = 10

            if add_day and day:
                add_day = str(add_day)
                day = str(day)
                add_day = datetime(year=int(add_day[0:4]), month=int(
                    add_day[4:6]), day=int(add_day[6:8]))
                day = datetime(year=int(day[0:4]), month=int(
                    day[4:6]), day=int(day[6:8]))
                interval_days = (day - add_day).days
            else:
                interval_days = 0

            interval_week = interval_days // 7

            wiki_for_userprofile = "空的空的"
            if wiki in featureEncodeHashInfo["wiki"]["hash_info"]:
                wiki_for_userprofile = wiki

            if topic_id not in featureEncodeHashInfo["topic_id"]["hash_info"]:
                topic_id = "空的空的"

            if topic_cate not in featureEncodeHashInfo["topic_cate"]["hash_info"]:
                topic_cate = "空的空的"

            if cate not in featureEncodeHashInfo["cate"]["hash_info"]:
                cate = "空的空的"

            if subcate not in featureEncodeHashInfo["subcate"]["hash_info"]:
                subcate = "空的空的"

            if author_type not in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                author_type = "空的空的"

            if author_city not in featureEncodeHashInfo["user_city"]["hash_info"]:
                author_city = "空的空的"

            if author_gender not in featureEncodeHashInfo["gender"]["hash_info"]:
                author_gender = "空的空的"

            author_uid_for_rerank = str(author_uid)

            item_like_7day = 0 if item_like_7day is None else float(item_like_7day)
            item_comment_7day = 0 if item_comment_7day is None else float(item_comment_7day)
            item_favorite_7day = 0 if item_favorite_7day is None else float(item_favorite_7day)
            item_click_7day = 0 if item_click_7day is None else float(item_click_7day)
            item_exp_7day = 0 if item_exp_7day is None else float(item_exp_7day)

            item_like_30day = 0 if item_like_30day is None else float(item_like_30day)
            item_comment_30day = 0 if item_comment_30day is None else float(item_comment_30day)
            item_favorite_30day = 0 if item_favorite_30day is None else float(item_favorite_30day)
            item_click_30day = 0 if item_click_30day is None else float(item_click_30day)
            item_exp_30day = 0 if item_exp_30day is None else float(item_exp_30day)

            author_becomment = 0 if author_becomment is None else float(author_becomment)
            author_befollow = 0 if author_befollow is None else float(author_befollow)
            author_belike = 0 if author_belike is None else float(author_belike)
            author_befavorite = 0 if author_befavorite is None else float(author_befavorite)

            hashValidClickObjFeature[doc_id] = {
                "doc_id" : doc_id,
                "favorite_num" : favorite_num,
                "like_num" : like_num,
                "comment_num" : comment_num,
                "score_level" : score_level,
                "interval_week" : interval_week,
                "wiki" : featureEncodeHashInfo["wiki"]["hash_info"][wiki_for_userprofile],
                "obj_type" : featureEncodeHashInfo["obj_type"]["hash_info"][obj_type_for_rerank],
                "topic_id" : featureEncodeHashInfo["topic_id"]["hash_info"][topic_id],
                "topic_cate" : featureEncodeHashInfo["topic_cate"]["hash_info"][topic_cate],
                "cate" : featureEncodeHashInfo["cate"]["hash_info"][cate],
                "subcate" : featureEncodeHashInfo["subcate"]["hash_info"][subcate],
                "author_type" : featureEncodeHashInfo["user_identity_type"]["hash_info"][author_type],
                "author_city" : featureEncodeHashInfo["user_city"]["hash_info"][author_city],
                "author_gender" : featureEncodeHashInfo["gender"]["hash_info"][author_gender],
                "author_uid" : author_uid_for_rerank,
                "item_like_7day" : item_like_7day,
                "item_comment_7day" : item_comment_7day,
                "item_favorite_7day" : item_favorite_7day,
                "item_click_7day" : item_click_7day,
                "item_exp_7day" : item_exp_7day,
                "item_like_30day" : item_like_30day,
                "item_comment_30day" : item_comment_30day,
                "item_favorite_30day" : item_favorite_30day,
                "item_click_30day" : item_click_30day,
                "item_exp_30day" : item_exp_30day,
                "author_becomment" : author_becomment,
                "author_befollow" : author_befollow,
                "author_belike" : author_belike,
                "author_befavorite" : author_befavorite,
            }

        presto_valid_click_data_cursor.close()

        sql = '''
                with j as (select
                                query, search_request_id, uid, obj_id, day
                            from oss.hhz.dws_search_user_personalization_rerank_daily_report
                            where  day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not  null and obj_id is not null and relevance is not null and is_valid is not null and  (is_valid = 1 or relevance > 1)),
        k as (select * from (select obj_id, count(*) as num  from j group by obj_id) where num >= 5),
        l as (select j.*, row_number() over (partition by j.uid  order by j.day desc, j.obj_id desc) rank from j join k on j.obj_id = k.obj_id order by rank asc),
        m as (select uid, array_distinct(array_agg(obj_id)) as "obj_id_list" from l group by l.uid),
        n as (select * from (select uid, cardinality(obj_id_list) as "obj_id_list_len",  slice(obj_id_list, 1, 60) as "obj_id_list"  from m order by obj_id_list_len desc))
        select uid, obj_id_list from n
                ''' % (dayTmp + 1, timeday + 1)

        log.logger.info("presto_cursor 用户有效点击序列: " + sql)
        presto_user_click_seq_cursor = presto_con.cursor()
        log.logger.info("presto cursor 用户有效点击序列 完成")
        presto_user_click_seq_cursor.execute(sql)
        log.logger.info("presto 执行 用户有效点击序列 完成")
        log.logger.info("开始解析数据 用户有效点击序列")

        userValidClickDatas = presto_user_click_seq_cursor.fetchall()

        allUserValidClickIds = []

        hashUserValidClickSeq = {}
        for userValidClickData in userValidClickDatas:
            uidStr = str(userValidClickData[0])
            doc_id_list = eval(str(userValidClickData[1]))

            if len(doc_id_list) > 0:
                hashUserValidClickSeq[uidStr] = {
                    "doc_id_seq": [],
                    "favorite_num_seq": [],
                    "like_num_seq": [],
                    "comment_num_seq": [],
                    "score_level_seq": [],
                    "interval_week_seq": [],
                    "wiki_seq": [],
                    "obj_type_seq": [],
                    "topic_id_seq": [],
                    "topic_cate_seq": [],
                    "cate_seq": [],
                    "subcate_seq": [],
                    "author_type_seq": [],
                    "author_city_seq": [],
                    "author_gender_seq": [],
                    "author_uid_seq": [],
                    "item_like_7day_seq": [],
                    "item_comment_7day_seq": [],
                    "item_favorite_7day_seq": [],
                    "item_click_7day_seq": [],
                    "item_exp_7day_seq": [],
                    "item_like_30day_seq": [],
                    "item_comment_30day_seq": [],
                    "item_favorite_30day_seq": [],
                    "item_click_30day_seq": [],
                    "item_exp_30day_seq": [],
                    "author_becomment_seq": [],
                    "author_befollow_seq": [],
                    "author_belike_seq": [],
                    "author_befavorite_seq": [],
                }

                for doc_id in doc_id_list:
                    if doc_id in hashValidClickObjFeature:
                        objFeature = hashValidClickObjFeature[doc_id]

                        hashUserValidClickSeq[uidStr]["doc_id_seq"].append(objFeature["doc_id"])
                        hashUserValidClickSeq[uidStr]["favorite_num_seq"].append(objFeature["favorite_num"])
                        hashUserValidClickSeq[uidStr]["like_num_seq"].append(objFeature["like_num"])
                        hashUserValidClickSeq[uidStr]["comment_num_seq"].append(objFeature["comment_num"])
                        hashUserValidClickSeq[uidStr]["score_level_seq"].append(objFeature["score_level"])
                        hashUserValidClickSeq[uidStr]["interval_week_seq"].append(objFeature["interval_week"])
                        hashUserValidClickSeq[uidStr]["wiki_seq"].append(objFeature["wiki"])
                        hashUserValidClickSeq[uidStr]["obj_type_seq"].append(objFeature["obj_type"])
                        hashUserValidClickSeq[uidStr]["topic_id_seq"].append(objFeature["topic_id"])
                        hashUserValidClickSeq[uidStr]["topic_cate_seq"].append(objFeature["topic_cate"])
                        hashUserValidClickSeq[uidStr]["cate_seq"].append(objFeature["cate"])
                        hashUserValidClickSeq[uidStr]["subcate_seq"].append(objFeature["subcate"])
                        hashUserValidClickSeq[uidStr]["author_type_seq"].append(objFeature["author_type"])
                        hashUserValidClickSeq[uidStr]["author_city_seq"].append(objFeature["author_city"])
                        hashUserValidClickSeq[uidStr]["author_gender_seq"].append(objFeature["author_gender"])
                        hashUserValidClickSeq[uidStr]["author_uid_seq"].append(objFeature["author_uid"])
                        hashUserValidClickSeq[uidStr]["item_like_7day_seq"].append(objFeature["item_like_7day"])
                        hashUserValidClickSeq[uidStr]["item_comment_7day_seq"].append(objFeature["item_comment_7day"])
                        hashUserValidClickSeq[uidStr]["item_favorite_7day_seq"].append(objFeature["item_favorite_7day"])
                        hashUserValidClickSeq[uidStr]["item_click_7day_seq"].append(objFeature["item_click_7day"])
                        hashUserValidClickSeq[uidStr]["item_exp_7day_seq"].append(objFeature["item_exp_7day"])
                        hashUserValidClickSeq[uidStr]["item_like_30day_seq"].append(objFeature["item_like_30day"])
                        hashUserValidClickSeq[uidStr]["item_comment_30day_seq"].append(objFeature["item_comment_30day"])
                        hashUserValidClickSeq[uidStr]["item_favorite_30day_seq"].append(objFeature["item_favorite_30day"])
                        hashUserValidClickSeq[uidStr]["item_click_30day_seq"].append(objFeature["item_click_30day"])
                        hashUserValidClickSeq[uidStr]["item_exp_30day_seq"].append(objFeature["item_exp_30day"])
                        hashUserValidClickSeq[uidStr]["author_becomment_seq"].append(objFeature["author_becomment"])
                        hashUserValidClickSeq[uidStr]["author_befollow_seq"].append(objFeature["author_befollow"])
                        hashUserValidClickSeq[uidStr]["author_belike_seq"].append(objFeature["author_belike"])
                        hashUserValidClickSeq[uidStr]["author_befavorite_seq"].append(objFeature["author_befavorite"])

            allUserValidClickIds.extend(doc_id_list)

        presto_cursor.close()

        # 获取 用户 历史搜索词
        sql = '''
                        with j as (select
                                query, search_request_id, uid, day
                            from oss.hhz.dws_search_user_personalization_rerank_daily_report
                            where  day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not  null and obj_id is not null and relevance is not null and is_valid is not null  group by query, search_request_id, uid, day),
        k as (select * from (select query, count(*) as num from j group by query) where num >= 31),
        l as (select j.*, row_number() over (partition by j.uid  order by j.day desc, j.query desc) rank from j join k on j.query = k.query order by rank asc),
        m as (select uid, array_distinct(array_agg(query)) as "query_list" from l group by l.uid),
        n as (select * from (select uid, cardinality(query_list) as "query_list_len",  slice(query_list, 1, 70) as "query_list"  from m )  order by query_list_len desc)
        select uid, query_list from n
                        ''' % (dayTmp + 1, timeday + 1)

        log.logger.info("presto_cursor 用户搜索历史序列: " + sql)
        presto_user_search_history_seq_cursor = presto_con.cursor()
        log.logger.info("presto cursor 用户搜索历史序列 完成")
        presto_user_search_history_seq_cursor.execute(sql)
        log.logger.info("presto 执行 用户搜索历史序列 完成")
        log.logger.info("开始解析数据 用户搜索历史序列")

        userSearchHistorySeqs = presto_user_search_history_seq_cursor.fetchall()
        hashUserSearchHistory = {}

        for userSearchHistorySeq in userSearchHistorySeqs:
            uid = userSearchHistorySeq[0]

            query_list = eval(str(userSearchHistorySeq[1]))

            if len(query_list) > 0 :
                hashUserSearchHistory[int(uid)] = query_list
                for queryItem in query_list:
                    queryItem = queryItem.strip()

                    if len(queryItem) > 0 and queryItem not in featureEncodeHashInfo["search_word"]["hash_info"]:
                        featureEncodeHashInfo["search_word"]["val"].append(queryItem)
                        index = len(featureEncodeHashInfo["search_word"]["val"])

                        featureEncodeHashInfo["search_word"]["index"].append(index)
                        featureEncodeHashInfo["search_word"]["hash_info"][queryItem] = index

        presto_user_search_history_seq_cursor.close()


        for query, click_data_format in  click_datas.items():
            for item in click_data_format["positive_list"]:
                docId = str(item[0])
                uid = str(item[13])

                doc_id_seq = []
                favorite_num_seq = []
                like_num_seq = []
                comment_num_seq = []
                score_level_seq = []
                interval_week_seq = []
                wiki_seq = []
                obj_type_seq = []
                topic_id_seq = []
                topic_cate_seq = []
                cate_seq = []
                subcate_seq = []
                author_type_seq = []
                author_city_seq = []
                author_gender_seq = []
                author_uid_seq = []
                item_like_7day_seq = []
                item_comment_7day_seq = []
                item_favorite_7day_seq = []
                item_click_7day_seq = []
                item_exp_7day_seq = []
                item_like_30day_seq = []
                item_comment_30day_seq = []
                item_favorite_30day_seq = []
                item_click_30day_seq = []
                item_exp_30day_seq = []
                author_becomment_seq = []
                author_befollow_seq = []
                author_belike_seq = []
                author_befavorite_seq = []

                isDeleteValidClickSeq = 0
                isDeleteSearchSeq = 0
                if uid in hashUserValidClickSeq:
                    uidStr = str(uid)

                    doc_id_seq = list(hashUserValidClickSeq[uidStr]["doc_id_seq"])
                    favorite_num_seq = list(hashUserValidClickSeq[uidStr]["favorite_num_seq"])
                    like_num_seq = list(hashUserValidClickSeq[uidStr]["like_num_seq"])
                    comment_num_seq = list(hashUserValidClickSeq[uidStr]["comment_num_seq"])
                    score_level_seq = list(hashUserValidClickSeq[uidStr]["score_level_seq"])
                    interval_week_seq = list(hashUserValidClickSeq[uidStr]["interval_week_seq"])
                    wiki_seq = list(hashUserValidClickSeq[uidStr]["wiki_seq"])
                    obj_type_seq = list(hashUserValidClickSeq[uidStr]["obj_type_seq"])
                    topic_id_seq = list(hashUserValidClickSeq[uidStr]["topic_id_seq"])
                    topic_cate_seq = list(hashUserValidClickSeq[uidStr]["topic_cate_seq"])
                    cate_seq = list(hashUserValidClickSeq[uidStr]["cate_seq"])
                    subcate_seq = list(hashUserValidClickSeq[uidStr]["subcate_seq"])
                    author_type_seq = list(hashUserValidClickSeq[uidStr]["author_type_seq"])
                    author_city_seq = list(hashUserValidClickSeq[uidStr]["author_city_seq"])
                    author_gender_seq = list(hashUserValidClickSeq[uidStr]["author_gender_seq"])
                    author_uid_seq = list(hashUserValidClickSeq[uidStr]["author_uid_seq"])
                    item_like_7day_seq = list(hashUserValidClickSeq[uidStr]["item_like_7day_seq"])
                    item_comment_7day_seq = list(hashUserValidClickSeq[uidStr]["item_comment_7day_seq"])
                    item_favorite_7day_seq = list(hashUserValidClickSeq[uidStr]["item_favorite_7day_seq"])
                    item_click_7day_seq = list(hashUserValidClickSeq[uidStr]["item_click_7day_seq"])
                    item_exp_7day_seq = list(hashUserValidClickSeq[uidStr]["item_exp_7day_seq"])
                    item_like_30day_seq = list(hashUserValidClickSeq[uidStr]["item_like_30day_seq"])
                    item_comment_30day_seq = list(hashUserValidClickSeq[uidStr]["item_comment_30day_seq"])
                    item_favorite_30day_seq = list(hashUserValidClickSeq[uidStr]["item_favorite_30day_seq"])
                    item_click_30day_seq = list(hashUserValidClickSeq[uidStr]["item_click_30day_seq"])
                    item_exp_30day_seq = list(hashUserValidClickSeq[uidStr]["item_exp_30day_seq"])
                    author_becomment_seq = list(hashUserValidClickSeq[uidStr]["author_becomment_seq"])
                    author_befollow_seq = list(hashUserValidClickSeq[uidStr]["author_befollow_seq"])
                    author_belike_seq = list(hashUserValidClickSeq[uidStr]["author_belike_seq"])
                    author_befavorite_seq = list(hashUserValidClickSeq[uidStr]["author_befavorite_seq"])

                    if docId in doc_id_seq:
                        isDeleteValidClickSeq = 1
                        delIndex = doc_id_seq.index(docId)

                        del doc_id_seq[delIndex]
                        del favorite_num_seq[delIndex]
                        del like_num_seq[delIndex]
                        del comment_num_seq[delIndex]
                        del score_level_seq[delIndex]
                        del interval_week_seq[delIndex]
                        del wiki_seq[delIndex]
                        del obj_type_seq[delIndex]
                        del topic_id_seq[delIndex]
                        del topic_cate_seq[delIndex]
                        del cate_seq[delIndex]
                        del subcate_seq[delIndex]
                        del author_type_seq[delIndex]
                        del author_city_seq[delIndex]
                        del author_gender_seq[delIndex]
                        del author_uid_seq[delIndex]
                        del item_like_7day_seq[delIndex]
                        del item_comment_7day_seq[delIndex]
                        del item_favorite_7day_seq[delIndex]
                        del item_click_7day_seq[delIndex]
                        del item_exp_7day_seq[delIndex]
                        del item_like_30day_seq[delIndex]
                        del item_comment_30day_seq[delIndex]
                        del item_favorite_30day_seq[delIndex]
                        del item_click_30day_seq[delIndex]
                        del item_exp_30day_seq[delIndex]
                        del author_becomment_seq[delIndex]
                        del author_befollow_seq[delIndex]
                        del author_belike_seq[delIndex]
                        del author_befavorite_seq[delIndex]

                item.append(doc_id_seq)
                item.append(favorite_num_seq)
                item.append(like_num_seq)
                item.append(comment_num_seq)
                item.append(score_level_seq)
                item.append(interval_week_seq)
                item.append(wiki_seq)
                item.append(obj_type_seq)
                item.append(topic_id_seq)
                item.append(topic_cate_seq)
                item.append(cate_seq)
                item.append(subcate_seq)
                item.append(author_type_seq)
                item.append(author_city_seq)
                item.append(author_gender_seq)
                item.append(author_uid_seq)
                item.append(item_like_7day_seq)
                item.append(item_comment_7day_seq)
                item.append(item_favorite_7day_seq)
                item.append(item_click_7day_seq)
                item.append(item_exp_7day_seq)
                item.append(item_like_30day_seq)
                item.append(item_comment_30day_seq)
                item.append(item_favorite_30day_seq)
                item.append(item_click_30day_seq)
                item.append(item_exp_30day_seq)
                item.append(author_becomment_seq)
                item.append(author_befollow_seq)
                item.append(author_belike_seq)
                item.append(author_befavorite_seq)

                search_history_seq = []
                if int(uid) in hashUserSearchHistory:
                    search_history_seq = list(hashUserSearchHistory[int(uid)])

                    if query in search_history_seq:
                        isDeleteSearchSeq = 1
                        delIndex = search_history_seq.index(query)
                        del search_history_seq[delIndex]

                item.append(search_history_seq)
                item.append(isDeleteValidClickSeq)
                item.append(isDeleteSearchSeq)


            for item in click_data_format["negative_list"]:
                docId = str(item[0])
                uid = str(item[13])

                doc_id_seq = []
                favorite_num_seq = []
                like_num_seq = []
                comment_num_seq = []
                score_level_seq = []
                interval_week_seq = []
                wiki_seq = []
                obj_type_seq = []
                topic_id_seq = []
                topic_cate_seq = []
                cate_seq = []
                subcate_seq = []
                author_type_seq = []
                author_city_seq = []
                author_gender_seq = []
                author_uid_seq = []
                item_like_7day_seq = []
                item_comment_7day_seq = []
                item_favorite_7day_seq = []
                item_click_7day_seq = []
                item_exp_7day_seq = []
                item_like_30day_seq = []
                item_comment_30day_seq = []
                item_favorite_30day_seq = []
                item_click_30day_seq = []
                item_exp_30day_seq = []
                author_becomment_seq = []
                author_befollow_seq = []
                author_belike_seq = []
                author_befavorite_seq = []

                isDeleteValidClickSeq = 0
                isDeleteSearchSeq = 0
                if uid in hashUserValidClickSeq:
                    uidStr = str(uid)

                    doc_id_seq = list(hashUserValidClickSeq[uidStr]["doc_id_seq"])
                    favorite_num_seq = list(hashUserValidClickSeq[uidStr]["favorite_num_seq"])
                    like_num_seq = list(hashUserValidClickSeq[uidStr]["like_num_seq"])
                    comment_num_seq = list(hashUserValidClickSeq[uidStr]["comment_num_seq"])
                    score_level_seq = list(hashUserValidClickSeq[uidStr]["score_level_seq"])
                    interval_week_seq = list(hashUserValidClickSeq[uidStr]["interval_week_seq"])
                    wiki_seq = list(hashUserValidClickSeq[uidStr]["wiki_seq"])
                    obj_type_seq = list(hashUserValidClickSeq[uidStr]["obj_type_seq"])
                    topic_id_seq = list(hashUserValidClickSeq[uidStr]["topic_id_seq"])
                    topic_cate_seq = list(hashUserValidClickSeq[uidStr]["topic_cate_seq"])
                    cate_seq = list(hashUserValidClickSeq[uidStr]["cate_seq"])
                    subcate_seq = list(hashUserValidClickSeq[uidStr]["subcate_seq"])
                    author_type_seq = list(hashUserValidClickSeq[uidStr]["author_type_seq"])
                    author_city_seq = list(hashUserValidClickSeq[uidStr]["author_city_seq"])
                    author_gender_seq = list(hashUserValidClickSeq[uidStr]["author_gender_seq"])
                    author_uid_seq = list(hashUserValidClickSeq[uidStr]["author_uid_seq"])
                    item_like_7day_seq = list(hashUserValidClickSeq[uidStr]["item_like_7day_seq"])
                    item_comment_7day_seq = list(hashUserValidClickSeq[uidStr]["item_comment_7day_seq"])
                    item_favorite_7day_seq = list(hashUserValidClickSeq[uidStr]["item_favorite_7day_seq"])
                    item_click_7day_seq = list(hashUserValidClickSeq[uidStr]["item_click_7day_seq"])
                    item_exp_7day_seq = list(hashUserValidClickSeq[uidStr]["item_exp_7day_seq"])
                    item_like_30day_seq = list(hashUserValidClickSeq[uidStr]["item_like_30day_seq"])
                    item_comment_30day_seq = list(hashUserValidClickSeq[uidStr]["item_comment_30day_seq"])
                    item_favorite_30day_seq = list(hashUserValidClickSeq[uidStr]["item_favorite_30day_seq"])
                    item_click_30day_seq = list(hashUserValidClickSeq[uidStr]["item_click_30day_seq"])
                    item_exp_30day_seq = list(hashUserValidClickSeq[uidStr]["item_exp_30day_seq"])
                    author_becomment_seq = list(hashUserValidClickSeq[uidStr]["author_becomment_seq"])
                    author_befollow_seq = list(hashUserValidClickSeq[uidStr]["author_befollow_seq"])
                    author_belike_seq = list(hashUserValidClickSeq[uidStr]["author_belike_seq"])
                    author_befavorite_seq = list(hashUserValidClickSeq[uidStr]["author_befavorite_seq"])

                    if docId in doc_id_seq:
                        isDeleteValidClickSeq = 1
                        delIndex = doc_id_seq.index(docId)

                        del doc_id_seq[delIndex]
                        del favorite_num_seq[delIndex]
                        del like_num_seq[delIndex]
                        del comment_num_seq[delIndex]
                        del score_level_seq[delIndex]
                        del interval_week_seq[delIndex]
                        del wiki_seq[delIndex]
                        del obj_type_seq[delIndex]
                        del topic_id_seq[delIndex]
                        del topic_cate_seq[delIndex]
                        del cate_seq[delIndex]
                        del subcate_seq[delIndex]
                        del author_type_seq[delIndex]
                        del author_city_seq[delIndex]
                        del author_gender_seq[delIndex]
                        del author_uid_seq[delIndex]
                        del item_like_7day_seq[delIndex]
                        del item_comment_7day_seq[delIndex]
                        del item_favorite_7day_seq[delIndex]
                        del item_click_7day_seq[delIndex]
                        del item_exp_7day_seq[delIndex]
                        del item_like_30day_seq[delIndex]
                        del item_comment_30day_seq[delIndex]
                        del item_favorite_30day_seq[delIndex]
                        del item_click_30day_seq[delIndex]
                        del item_exp_30day_seq[delIndex]
                        del author_becomment_seq[delIndex]
                        del author_befollow_seq[delIndex]
                        del author_belike_seq[delIndex]
                        del author_befavorite_seq[delIndex]

                item.append(doc_id_seq)
                item.append(favorite_num_seq)
                item.append(like_num_seq)
                item.append(comment_num_seq)
                item.append(score_level_seq)
                item.append(interval_week_seq)
                item.append(wiki_seq)
                item.append(obj_type_seq)
                item.append(topic_id_seq)
                item.append(topic_cate_seq)
                item.append(cate_seq)
                item.append(subcate_seq)
                item.append(author_type_seq)
                item.append(author_city_seq)
                item.append(author_gender_seq)
                item.append(author_uid_seq)
                item.append(item_like_7day_seq)
                item.append(item_comment_7day_seq)
                item.append(item_favorite_7day_seq)
                item.append(item_click_7day_seq)
                item.append(item_exp_7day_seq)
                item.append(item_like_30day_seq)
                item.append(item_comment_30day_seq)
                item.append(item_favorite_30day_seq)
                item.append(item_click_30day_seq)
                item.append(item_exp_30day_seq)
                item.append(author_becomment_seq)
                item.append(author_befollow_seq)
                item.append(author_belike_seq)
                item.append(author_befavorite_seq)

                search_history_seq = []
                if int(uid) in hashUserSearchHistory:
                    search_history_seq = list(hashUserSearchHistory[int(uid)])

                    if query in search_history_seq:
                        isDeleteSearchSeq = 1
                        delIndex = search_history_seq.index(query)
                        del search_history_seq[delIndex]

                item.append(search_history_seq)

                item.append(isDeleteValidClickSeq)
                item.append(isDeleteSearchSeq)


        f = open(featureEncodeJsonFile, 'w')
        f.write(json.dumps(featureEncodeHashInfo))
        f.close()

        needDeleteKeys = []
        for keyword, click_data_format in click_datas.items():
            if len(click_data_format["positive_list"]) == 0 :
                needDeleteKeys.append(keyword)

        for needDeleteKey in needDeleteKeys:
            del click_datas[needDeleteKey]

        # 处理 搜索词 下 正样本 永远不会成为负样本
        sql = '''
                with a as (select
                        query as "搜索词",
                        obj_id as "内容id",
                        max(relevance) as relevance,
                        day as "搜索行为发生的日期",
                        max(is_valid) as is_valid
                    from oss.hhz.dws_search_user_personalization_rerank_daily_report
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null
                    group by day,query,obj_id)
select * from a where (relevance = 1 and is_valid = 1) or relevance > 1
                ''' % (timeday)

        log.logger.info("presto_cursor 搜索词下内容id被点击的情况: " + sql)
        presto_search_keyword_cursor = presto_con.cursor()
        log.logger.info("presto cursor 搜索词下内容id被点击的情况 完成")
        presto_search_keyword_cursor.execute(sql)
        log.logger.info("presto 执行 搜索词下内容id被点击的情况 完成")
        log.logger.info("开始解析数据 搜索词下内容id被点击的情况")

        hashQueryObjPositiveData = {}
        for search_keyword_info in presto_search_keyword_cursor.fetchall():
            query = search_keyword_info[0].strip()

            doc_id = search_keyword_info[1].strip()

            if len(doc_id) == 0:
                continue

            relevance = search_keyword_info[2]
            is_valid = search_keyword_info[4]

            if query not in hashQueryObjPositiveData:
                hashQueryObjPositiveData[query] = []

            hashQueryObjPositiveData[query].append(doc_id)

        presto_search_keyword_cursor.close()


        return click_datas, corpus_obj_ids_set, hashQueryObjPositiveData

    def get_corpus(self, obj_ids_list):
        from data_utils.Thread.dataProcessFromDb import dataProcessFromDb
        from data_utils.Thread.dataProcessInsert import dataProcessInsert
        from multiprocessing import Queue
        import multiprocessing
        '''
        Author: xiaoyichao
        param {*} self
        param {*} obj_ids_list
        Description: 获取obj_id 和 文本 组成的字典
        '''

        def get_data_by_thread(obj_ids, content_type, corpus):
            objNum = len(obj_ids)
            globalTaskNum = multiprocessing.Value('d', objNum)

            data_queue = Queue(100000)  # 存放解析数据的queue

            objIdQueue = Queue(100000)

            objType = ""
            if content_type == 1:
                objType = "blank"
            elif content_type == 2:
                objType = "note"
            elif content_type == 3:
                objType = "article"
            elif content_type == 4:
                objType = "guide"

            threadNum = 5
            thread_insert_id = 'insert_queue_1'
            threadInsert = dataProcessInsert(thread_insert_id, objIdQueue, content_type, obj_ids, threadNum)  # 启动线程
            threadInsert.start()  # 启动线程

            # 初始化请求线程
            get_info_from_db_threads = []
            get_info_name_list = ['get_info_from_db_' + str(i) for i in range(threadNum)]

            for thread_id in get_info_name_list:
                thread = dataProcessFromDb(thread_id, objIdQueue, data_queue, objNum, content_type, objType,
                                           globalTaskNum)  # 启动线程
                thread.start()  # 启动线程
                get_info_from_db_threads.append(thread)

            NoneNum = 0
            while True:
                objid_title_remark_dict = data_queue.get(True)
                if objid_title_remark_dict is None:
                    NoneNum = NoneNum + 1
                    if NoneNum == threadNum:
                        break
                    continue
                corpus.update(objid_title_remark_dict)

        corpus = {}
        # self.add_obj_pass_ids = []
        if len(obj_ids_list) > 0:
            # objid_content_type_dict = self.get_objid_content_type_dict(obj_ids_list)
            content_type_ids_dict = Tool.get_content_type_ids_dict(obj_ids_list)
            # {content_type:1是文章，2是note，3是整屋，4是指南}
            for content_type, obj_ids in content_type_ids_dict.items():
                if len(obj_ids) > 0:
                    get_data_by_thread(obj_ids, content_type, corpus)

        return corpus

    def get_samples(self, click_datas, corpus, max_relevance=3, pos_neg_ration=4, hashQueryObjPositiveData = None, max_len=32):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成samples
        '''
        corpus_obj_ids_set = set(corpus.keys())

        hashFilterQueryObjPositiveData = {}
        for query, queryObjsPositive in hashQueryObjPositiveData.items():
            for objId in queryObjsPositive:
                if objId in corpus_obj_ids_set:
                    # 这个case 会把那些同义词的结果都变成负样本
                    title = corpus[objId]["title"]
                    remark = corpus[objId]["remark"]

                    if query in title[:max_len] or query in remark[:max_len]:
                        if query not in hashFilterQueryObjPositiveData:
                            hashFilterQueryObjPositiveData[query] = set()

                        hashFilterQueryObjPositiveData[query].add(objId)


        samples = []

        eval_samples_classify = {}

        allPositiveNum = 0
        allNegativeSamples = []

        for query, click_data in click_datas.items():

            query = click_data['query']
            if len(click_data['positive_list']) > 0 and len(click_data['negative_list']) > 0:
                # 正样本 为 空的情况下 没必要训练该搜索词
                pos_cnt_tmp = 0
                # 获取 正样本 数量
                for pos_id_rele_data in click_data['positive_list']:
                    objId = pos_id_rele_data[0]

                    if objId in corpus_obj_ids_set:
                        # 这个case 会把那些同义词的结果都变成负样本
                        title = corpus[objId]["title"]
                        remark = corpus[objId]["remark"]

                        if query in title[:max_len] or query in remark[:max_len]:
                            pos_cnt_tmp += 1

                # 正样本为空 没必要训练
                if pos_cnt_tmp == 0:
                    continue

                neg_cnt = 0
                pos_cnt = 0
                for pos_id_rele_data in click_data['positive_list']:
                    if pos_id_rele_data[0] in corpus_obj_ids_set:
                        unique_str = str(pos_id_rele_data[13]) + "_" + query

                        if unique_str not in eval_samples_classify:
                            eval_samples_classify[unique_str] = []

                        # 这个case 会把那些同义词的结果都变成负样本
                        title = corpus[pos_id_rele_data[0]]["title"]
                        remark = corpus[pos_id_rele_data[0]]["remark"]
                        title_tf_num = get_tf(query, title, have_jiebad=False,
                                              max_len=128)  # 这个max_len不需要跟bert的max_len一样
                        remark_tf_num = get_tf(query, remark, have_jiebad=False, max_len=128)
                        if query in title[:max_len] or query in remark[:max_len]:
                            pos_cnt += 1
                            label = pos_id_rele_data[1]
                            data4train = [pos_id_rele_data[0], label, query, title, remark, title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                            for d in pos_id_rele_data[2:]:
                                data4train.append(d)
                            samples.append(data4train)

                            eval_samples_classify[unique_str].append(data4train)
                            # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                        else:
                            # if neg_cnt < pos_cnt*pos_neg_ration:
                            neg_cnt += 1
                            label = 0
                            data4train = [pos_id_rele_data[0], label, query, title, remark, title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                            for d in pos_id_rele_data[2:]:
                                data4train.append(d)
                            # samples.append(data4train)

                            allNegativeSamples.append(data4train)

                            # eval_samples_classify[unique_str].append(data4train)
                            # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                allPositiveNum += pos_cnt

                random.shuffle(click_data['negative_list'])
                for neg_id_rele_data in click_data['negative_list']:
                    if neg_id_rele_data[0] in corpus_obj_ids_set:
                        # if neg_cnt < pos_cnt * pos_neg_ration:
                        unique_str = str(neg_id_rele_data[13]) + "_" + query

                        if unique_str not in eval_samples_classify:
                            eval_samples_classify[unique_str] = []

                        neg_cnt += 1
                        # label = float(neg_id_rele[1]/max_relevance)
                        label = 0
                        title = corpus[neg_id_rele_data[0]]["title"]
                        remark = corpus[neg_id_rele_data[0]]["remark"]
                        title_tf_num = get_tf(query, title, have_jiebad=False,
                                              max_len=128)  # 这个max_len不需要跟bert的max_len一样
                        remark_tf_num = get_tf(query, remark, have_jiebad=False, max_len=128)

                        data4train = [neg_id_rele_data[0], label, query, title, remark, title_tf_num, remark_tf_num]
                        # doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type
                        for d in neg_id_rele_data[2:]:
                            data4train.append(d)
                        # samples.append(data4train)
                        if not (query in hashFilterQueryObjPositiveData and neg_id_rele_data[0] in hashFilterQueryObjPositiveData[
                                    query]):
                            allNegativeSamples.append(data4train)

                        # if neg_cnt < pos_cnt * pos_neg_ration:
                        #     eval_samples_classify[unique_str].append(data4train)
                        # else:
                        #     break

        random.shuffle(allNegativeSamples)
        log.logger.info("allNegativeSamples 样本数" + str(len(allNegativeSamples)))
        log.logger.info("allPositiveNum 数" + str(allPositiveNum))
        for i in range(4 * allPositiveNum):
            samples.append(allNegativeSamples[i])

        random.shuffle(samples)

        with open(featureEncodeJsonFile, 'r') as load_f:
            featureEncodeHashInfo = json.load(load_f)

        for sample in samples:
            docId = sample[0]

            qyery = sample[2].strip()
            if len(qyery) > 0 and qyery not in featureEncodeHashInfo["search_word"]["hash_info"]:
                featureEncodeHashInfo["search_word"]["val"].append(qyery)
                index = len(featureEncodeHashInfo["search_word"]["val"])

                featureEncodeHashInfo["search_word"]["index"].append(index)
                featureEncodeHashInfo["search_word"]["hash_info"][qyery] = index
            sample.append(featureEncodeHashInfo["search_word"]["hash_info"][qyery])

            uidStr = str(sample[18])

            if len(docId) > 0 and  docId not in featureEncodeHashInfo["obj_id"]["hash_info"] :
                featureEncodeHashInfo["obj_id"]["val"].append(docId)
                index = len(featureEncodeHashInfo["obj_id"]["val"])

                featureEncodeHashInfo["obj_id"]["index"].append(index)
                featureEncodeHashInfo["obj_id"]["hash_info"][docId] = index
            sample[0] = featureEncodeHashInfo["obj_id"]["hash_info"][docId]

            if len(uidStr) > 0 and uidStr not in featureEncodeHashInfo["uid"]["hash_info"]:
                featureEncodeHashInfo["uid"]["val"].append(uidStr)
                index = len(featureEncodeHashInfo["uid"]["val"])

                featureEncodeHashInfo["uid"]["index"].append(index)
                featureEncodeHashInfo["uid"]["hash_info"][uidStr] = index
            sample[18] = featureEncodeHashInfo["uid"]["hash_info"][uidStr]

            authorUidStr = str(sample[23])
            if len(authorUidStr) > 0 and authorUidStr not in featureEncodeHashInfo["uid"]["hash_info"]:
                featureEncodeHashInfo["uid"]["val"].append(authorUidStr)
                index = len(featureEncodeHashInfo["uid"]["val"])

                featureEncodeHashInfo["uid"]["index"].append(index)
                featureEncodeHashInfo["uid"]["hash_info"][authorUidStr] = index
            sample[23] = featureEncodeHashInfo["uid"]["hash_info"][authorUidStr]

            validClickDocIdSeq = sample[47]
            validClickDocIdTransSeq = []
            for docIndex, docId in enumerate(validClickDocIdSeq):
                if len(docId) > 0 and docId not in featureEncodeHashInfo["obj_id"]["hash_info"]:
                    featureEncodeHashInfo["obj_id"]["val"].append(docId)
                    index = len(featureEncodeHashInfo["obj_id"]["val"])

                    featureEncodeHashInfo["obj_id"]["index"].append(index)
                    featureEncodeHashInfo["obj_id"]["hash_info"][docId] = index

                validClickDocIdTransSeq.append(featureEncodeHashInfo["obj_id"]["hash_info"][docId])
            sample[47] = validClickDocIdTransSeq

            validClickDocAuthorIdSeq = sample[62]
            validClickDocAuthorIdTransSeq = []
            for uidIndex, authorUid in enumerate(validClickDocAuthorIdSeq):
                authorUidStr = str(authorUid)

                if len(authorUidStr) > 0 and authorUidStr not in featureEncodeHashInfo["uid"]["hash_info"]:
                    featureEncodeHashInfo["uid"]["val"].append(authorUidStr)
                    index = len(featureEncodeHashInfo["uid"]["val"])

                    featureEncodeHashInfo["uid"]["index"].append(index)
                    featureEncodeHashInfo["uid"]["hash_info"][authorUidStr] = index

                validClickDocAuthorIdTransSeq.append(featureEncodeHashInfo["uid"]["hash_info"][authorUidStr])
            sample[62] = validClickDocAuthorIdTransSeq

            searchWordHistorySeq = sample[77]
            searchWordHistoryTransSeq = []
            for uidIndex, searchWord in enumerate(searchWordHistorySeq):
                searchWord = searchWord.strip()

                if len(searchWord) > 0 and searchWord not in featureEncodeHashInfo["search_word"]["hash_info"]:
                    featureEncodeHashInfo["search_word"]["val"].append(searchWord)
                    index = len(featureEncodeHashInfo["search_word"]["val"])

                    featureEncodeHashInfo["search_word"]["index"].append(index)
                    featureEncodeHashInfo["search_word"]["hash_info"][searchWord] = index

                searchWordHistoryTransSeq.append(featureEncodeHashInfo["search_word"]["hash_info"][searchWord])
            sample[77] = searchWordHistoryTransSeq

        with open(featureEncodeJsonFile, 'w') as f:
            f.write(json.dumps(featureEncodeHashInfo))

        log.logger.info("samples 长度: " + str(len(samples)))

        # 过滤 没有 正样本 的 验证序列
        needDeleteKeys = []
        for unique_str, sample_list in eval_samples_classify.items():
            isHavePositive = 0

            for item in sample_list:
                if int(item[1]) > 0:
                    isHavePositive = 1
                    break

            if isHavePositive == 0:
                needDeleteKeys.append(unique_str)

        for needDeleteKey in needDeleteKeys:
            del eval_samples_classify[needDeleteKey]

        hashSampleData = {}
        for sample in samples:
            uidIndexSample = sample[18]
            queryIndexSample = sample[80]

            unique_sample_str = str(uidIndexSample) + "----------" + str(queryIndexSample)
            if unique_sample_str not in hashSampleData:
                hashSampleData[unique_sample_str] = []

            hashSampleData[unique_sample_str].append(sample)

        # 重新 处理每个用户 每个搜索词 的下的样本点击情况。 扩充验证集 否则会出现某些用户的搜索词下样本太少，过于简单，评分偏高
        # 构建 eval_samples_classify 验证集
        for uid_search_key, items in eval_samples_classify.items():
            uidIndex = items[0][18]
            queryIndex = items[0][80]

            uid_for_rerank = items[0][18]
            user_identity_type = items[0][13]
            gender = items[0][16]
            age = items[0][17]
            user_city = items[0][27]
            decoration_status = items[0][28]
            action_count_like = items[0][42]
            action_count_comment = items[0][43]
            action_count_favor = items[0][44]
            action_count_all_active = items[0][45]
            valid_click_doc_id_seq = items[0][47]
            valid_click_topic_id_seq = items[0][55]
            valid_click_topic_cate_seq = items[0][56]
            valid_click_cate_seq = items[0][57]
            valid_click_subcate_seq = items[0][58]
            valid_click_author_city_seq = items[0][60]
            valid_click_author_uid_seq = items[0][62]
            search_word_history_seq = items[0][77]

            hashUidQuery = {}
            unique_str = str(uidIndex) + "----------" + str(queryIndex)
            for item in items:
                if unique_str not in hashUidQuery:
                    hashUidQuery[unique_str] = set()
                hashUidQuery[unique_str].add(item[0])

            if unique_str in hashSampleData:
                sampleDatas = hashSampleData[unique_str]

                for sample in sampleDatas:
                    uidIndexSample = sample[18]
                    queryIndexSample = sample[80]
                    docIdIndexSample = sample[0]

                    unique_sample_str = str(uidIndexSample) + "----------" + str(queryIndexSample)
                    if unique_sample_str == unique_str:
                        if docIdIndexSample not in hashUidQuery[unique_str]:
                            sampleItem = list(sample)

                            sampleItem[18] = uid_for_rerank
                            sampleItem[13] = user_identity_type
                            sampleItem[16] = gender
                            sampleItem[17] = age
                            sampleItem[27] = user_city
                            sampleItem[28] = decoration_status
                            sampleItem[42] = action_count_like
                            sampleItem[43] = action_count_comment
                            sampleItem[44] = action_count_favor
                            sampleItem[45] = action_count_all_active
                            sampleItem[47] = valid_click_doc_id_seq
                            sampleItem[55] = valid_click_topic_id_seq
                            sampleItem[56] = valid_click_topic_cate_seq
                            sampleItem[57] = valid_click_cate_seq
                            sampleItem[58] = valid_click_subcate_seq
                            sampleItem[60] = valid_click_author_city_seq
                            sampleItem[62] = valid_click_author_uid_seq
                            sampleItem[77] = search_word_history_seq

                            sampleItem[1] = 0

                            items.append(sampleItem)

            random.shuffle(items)

        # 控制验证集中 某个 用户 在某个搜索词下 的 样本数量
        eval_samples_classify_return = {}
        for uid_search_key, items in eval_samples_classify.items():
            if len(items) >= 10:
                eval_samples_classify_return[uid_search_key] = items

        return samples, eval_samples_classify_return

def write_samples_pkl(samples, output_file):
    '''
    Author: xiaoyichao
    param {*}
    Description: 将samples数据写入pkl文件
    '''
    output_file_path_list = output_file.split("/")
    output_file_parent_path = "/".join(output_file_path_list[:-1])
    if not os.path.exists(output_file_parent_path):
        os.makedirs(output_file_parent_path)
    with open(output_file, 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(samples, fo)
    log.logger.info("数据写入%s文件" % str(output_file))


def read_samples_pkl(pkl_file):
    '''
    Author: xiaoyichao
    param {*}
    Description: 从pkl文件中读取出samples数据
    '''
    with open(pkl_file, 'rb') as f:     # 读取pkl文件数据
        samples = pickle.load(f, encoding='bytes')
    return samples


def get_before_day(before_day):
    '''
    Author: xiaoyichao
    param {*}
    Description: 获取几天前的日期
    '''
    import datetime
    today = datetime.date.today()
    before_days = datetime.timedelta(days=before_day)
    before_day = today-before_days
    return str(before_day)


def merge_data(pkl_file_list):
    '''
    Author: xiaoyichao
    param {*}
    Description: 合并数据，但是要考虑是否要对数据去重复,或者保存字典，然后去重复
    '''
    samples_sum = []
    for pkl_file in pkl_file_list:
        samples = read_samples_pkl(pkl_file)
        samples_sum.extend(samples)

import json, os

# 获取 特征 编码
def getHashData():
    hashUserFeature = {}
    hashObjFeature = {}

    if os.path.exists(fineRowHashDataJsonFile):
        with open(fineRowHashDataJsonFile, 'r') as load_f:
            fineRowHashData = json.load(load_f)

            hashUserFeature = fineRowHashData["hash_uid_feature"]
            hashObjFeature = fineRowHashData["hash_obj_id_feature"]

    return hashUserFeature, hashObjFeature

# 获取 特征 编码
def getFeaturesEncode():
    featureEncodeInfo = {}

    if os.path.exists(featureEncodeJsonFile):
        with open(featureEncodeJsonFile, 'r') as load_f:
            featureEncodeInfo = json.load(load_f)

    return featureEncodeInfo


def get_click_data():
    # before_day = get_before_day(1)
    # 获取 内容 id 对应 数字
    POS_NEG_RATION = 4  # 正负样本比例1:4
    max_len = 64
    max_relevance = 3
    corpus_obj_ids_set_sum = set()
    corpus_sum = {}

    clickdata = ClickData()
    for day in range(1, 2, 1):
        click_datas, corpus_obj_ids_set, hashQueryObjPositiveData = clickdata.get_click_datas4bert(day=day)

        corpus_obj_ids_set_add = corpus_obj_ids_set - corpus_obj_ids_set_sum
        #
        log.logger.info("len(click_datas): " + str(len(click_datas)))
        #
        corpus = clickdata.get_corpus(corpus_obj_ids_set_add)
        #
        corpus_sum.update(corpus)

        corpus_obj_ids_set_sum = corpus_obj_ids_set_sum | corpus_obj_ids_set
        log.logger.info("len(corpus_sum): " + str(len(corpus_sum)))
        click_format_datas, eval_classify_samples = clickdata.get_samples(click_datas = click_datas, corpus=corpus_sum, max_relevance=max_relevance, pos_neg_ration=POS_NEG_RATION, hashQueryObjPositiveData=hashQueryObjPositiveData, max_len=max_len)

        print("=======click_format_datas=========")
        num = 0
        for click_format_data in click_format_datas:
            print(click_format_data)

            num = num + 1
            if num == 2:
                break
        print("=======click_format_datas=========")

        pkl_file = os.path.join('/data/search_opt_model/topk_opt/rank_fine_row_userprofile_neg_v2/data_'+get_before_day(day)+'.pkl')
        write_samples_pkl(click_format_datas, pkl_file)

        npy_classify_file = os.path.join('/data/search_opt_model/topk_opt/rank_fine_row_userprofile_neg_eval_v2/eval_classify_' + get_before_day(
                                             day) + '.npy')

        hash_eval_classify_data = {
            "eval_classify_samples": eval_classify_samples
        }

        np.save(npy_classify_file, hash_eval_classify_data)

        log.logger.info("数据读取完成"+ str(day))
    log.logger.info("程序执行完毕")
    # send_msg("pkl文件数据 用户个性化重排 训练文件 生成完成")


if __name__ == '__main__':
    get_click_data()
