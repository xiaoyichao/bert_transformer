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
from presto_con.topresto import python_presto
import random
from common.common_log import Logger, get_log_name
from common.tool import Tool
import json

from common.send_message import send_msg
from data_utils.fine_row_userprofile_classify_neg_v2 import featureEncodeJsonFile, getHashData


current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

fineRowHashAllDataJsonFile = "/data/search_opt_model/topk_opt/fine_row_hash_data_neg_v2_all_data.json"

class BuildAllFeature(object):
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

    def build_all_feature(self, day):
        presto_con = self.presto.connect(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         password=self.password
                                         )
        log.logger.info("presto连接完成")

        sql = '''
        with k as (select  
    max(user_identity_type) user_identity_type,
    max(gender) gender,
    max(age) age, 
    uid, 
    max(user_city) user_city, 
    max(decoration_status) decoration_status,
  ln(1+max(action_count_like)) action_count_like,
  ln(1+max(action_count_comment)) action_count_comment, 
  ln(1+max(action_count_favor)) action_count_favor, 
  ln(1+max(action_count_all_active)) action_count_all_active,
  'a' flag
  from oss.hhz.dws_search_user_personalization_rerank_daily_report  where
 day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not  null and obj_id is not null and relevance is not null and is_valid is not null group by query, obj_id, day, uid),
l as (
    select 
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
     uid,
    user_identity_type,
    gender,
    age,
    city,
    decoration_status,
    round((ln(1+action_count_like) - l.min_action_count_like)/ (l.max_action_count_like - l.min_action_count_like), 4),
    round((ln(1+action_count_comment) - l.min_action_count_comment)/ (l.max_action_count_comment - l.min_action_count_comment), 4),
    round((ln(1+action_count_favor) - l.min_action_count_favor)/ (l.max_action_count_favor - l.min_action_count_favor), 4),
    round((ln(1+action_count_all_active) - l.min_action_count_all_active)/ (l.max_action_count_all_active - l.min_action_count_all_active), 4)
from oss.hhz.search_user_model_trait_daily,l where day  between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd}
''' % (day, day, day, day)

        log.logger.info("presto_cursor 获取用户特征: "+sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 获取用户特征 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 获取用户特征 完成")
        log.logger.info("开始解析数据 获取用户特征")

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

        hashUserFeature, _ = getHashData()

        # hashUserFeature = {}
        hashObjFeature = {}

        for info in fetchDatas:
            uid = 0 if info[0] is None else info[0]
            uid = int(uid)
            if uid == 0 or str(uid) in hashUserFeature:
                continue

            uidIndex = 0
            if str(uid) in featureEncodeHashInfo["uid"]["hash_info"]:
                uidIndex = featureEncodeHashInfo["uid"]["hash_info"][str(uid)]

            user_identity_type = info[1]
            user_identity_type = "" if user_identity_type is None else user_identity_type
            if len(user_identity_type) > 0 and user_identity_type  in featureEncodeHashInfo["user_identity_type"]["hash_info"]:
                user_identity_type = featureEncodeHashInfo["user_identity_type"]["hash_info"][user_identity_type]
            else:
                user_identity_type = 0

            gender = info[2]
            gender = "" if gender is None else gender
            if len(gender) > 0 and gender in featureEncodeHashInfo["gender"][
                "hash_info"]:
                gender = featureEncodeHashInfo["gender"]["hash_info"][gender]
            else:
                gender = 0

            age = info[3]
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

            city = info[4]
            city = "" if city is None else city
            if len(city) > 0 and city in featureEncodeHashInfo["user_city"][
                "hash_info"]:
                city = featureEncodeHashInfo["user_city"]["hash_info"][city]
            else:
                city = 0

            decoration_status = info[5]
            decoration_status = "" if decoration_status is None else decoration_status
            if len(decoration_status) > 0 and decoration_status in featureEncodeHashInfo["decoration_status"][
                "hash_info"]:
                decoration_status = featureEncodeHashInfo["decoration_status"]["hash_info"][decoration_status]
            else:
                decoration_status = 0

            action_count_like = info[6]
            action_count_like = 0 if action_count_like is None else float(action_count_like)

            action_count_comment = info[7]
            action_count_comment = 0 if action_count_comment is None else float(action_count_comment)

            action_count_favor = info[8]
            action_count_favor = 0 if action_count_favor is None else float(action_count_favor)

            action_count_all_active = info[9]
            action_count_all_active = 0 if action_count_all_active is None else float(action_count_all_active)

            userFeature = {
                "uid_for_rerank" : float(uidIndex),
                "user_identity_type" : user_identity_type,
                "gender" : gender,
                "age" : float(age_level),
                "user_city" : city,
                "decoration_status" : decoration_status,
                "action_count_like" : action_count_like,
                "action_count_comment" : action_count_comment,
                "action_count_favor" : action_count_favor,
                "action_count_all_active" : action_count_all_active,
                "valid_click_doc_id_seq" : [],
                "valid_click_topic_id_seq" : [],
                "valid_click_topic_cate_seq" : [],
                "valid_click_cate_seq" : [],
                "valid_click_subcate_seq" : [],
                "valid_click_author_city_seq" : [],
                "valid_click_author_uid_seq" : [],
                "search_word_history_seq" : [],
            }

            hashUserFeature[str(uid)] = userFeature

        presto_cursor.close()

        sql = '''
                with k as (select  
    ln(1+max(favorite_num)) favorite_num, 
    ln(1+max(like_num)) like_num, 
    ln(1+max(comment_num)) comment_num, 
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
      'a' flag
    from k
)
select 
obj_id,
round((ln(1+favorite_num) - l.min_favorite_num)/ (l.max_favorite_num - l.min_favorite_num), 4),
round((ln(1+like_num) - l.min_like_num)/ (l.max_like_num - l.min_like_num), 4),
round((ln(1+comment_num) - l.min_comment_num)/ (l.max_comment_num - l.min_comment_num), 4),
admin_score,
publish_day,
is_wiki,
obj_type,
topic_id,
topic_cate,
cate,
subcate,
author_id,
author_type,
author_city,
gender,
round((ln(1+item_like_7day) - l.min_item_like_7day)/ (l.max_item_like_7day - l.min_item_like_7day), 4),
round((ln(1+item_comment_7day) - l.min_item_comment_7day)/ (l.max_item_comment_7day - l.min_item_comment_7day), 4),
round((ln(1+item_favorite_7day) - l.min_item_favorite_7day)/ (l.max_item_favorite_7day - l.min_item_favorite_7day), 4),
round((ln(1+item_click_7day) - l.min_item_click_7day)/ (l.max_item_click_7day - l.min_item_click_7day), 4),
round((ln(1+item_exp_7day) - l.min_item_exp_7day)/ (l.max_item_exp_7day - l.min_item_exp_7day), 4),
round((ln(1+item_like_30day) - l.min_item_likek_30day)/ (l.max_item_likek_30day - l.min_item_likek_30day), 4),
round((ln(1+item_comment_30day) - l.min_item_comment_30day)/ (l.max_item_comment_30day - l.min_item_comment_30day), 4),
round((ln(1+item_favorite_30day) - l.min_item_favorite_30day)/ (l.max_item_favorite_30day - l.min_item_favorite_30day), 4),
round((ln(1+item_click_30day) - l.min_item_click_30day)/ (l.max_item_click_30day - l.min_item_click_30day), 4),
round((ln(1+item_exp_30day) - l.min_item_exp_30day)/ (l.max_item_exp_30day - l.min_item_exp_30day), 4),
round((ln(1+author_becomment) - l.min_author_becomment)/ (l.max_author_becomment - l.min_author_becomment), 4),
round((ln(1+author_befollow) - l.min_author_befollow)/ (l.max_author_befollow - l.min_author_befollow), 4),
round((ln(1+author_belike) - l.min_author_belike)/ (l.max_author_belike - l.min_author_belike), 4),
round((ln(1+author_befavorite) - l.min_author_befavorite)/ (l.max_author_befavorite - l.min_author_befavorite), 4),
day
from oss.hhz.search_obj_model_trait_daily,l where day  between  #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd}
        ''' % (day, day, day, day)

        log.logger.info("presto_cursor 获取内容特征: " + sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 获取内容特征 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 获取内容特征 完成")
        log.logger.info("开始解析数据 获取内容特征")

        fetchDatas = presto_cursor.fetchall()

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        obj_type_dict_keys = set(obj_type_dict.keys())

        for info in fetchDatas:
            objId = info[0]
            objId = "" if objId is None else objId
            if len(objId) == 0 or objId in hashObjFeature:
                continue

            objIdIndex = 0
            if objId in featureEncodeHashInfo["obj_id"][
                "hash_info"]:
                objIdIndex = featureEncodeHashInfo["obj_id"]["hash_info"][objId]

            favorite_num = info[1]
            favorite_num = 0 if favorite_num is None else float(favorite_num)

            like_num = info[2]
            like_num = 0 if like_num is None else float(like_num)

            comment_num = info[3]
            comment_num = 0 if comment_num is None else float(comment_num)

            score = info[4]
            score = 0 if score is None else int(score)

            obj_type_num = Tool.getTypeByObjId(objId)
            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict_keys else obj_type_dict[obj_type_num]
            obj_type = obj_type.strip()

            if obj_type == "note" or obj_type == "回答的note":
                if score < 60:
                    score = 60

            score_level = 0
            if score <= 10:
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

            search_day = info[30]
            add_day = info[5]
            if add_day and search_day:
                add_day = str(add_day)
                search_day = str(search_day)
                add_day = datetime(year=int(add_day[0:4]), month=int(
                    add_day[4:6]), day=int(add_day[6:8]))
                search_day = datetime(year=int(search_day[0:4]), month=int(
                    search_day[4:6]), day=int(search_day[6:8]))
                interval_days = (search_day - add_day).days
            else:
                interval_days = 0

            interval_week = interval_days // 7

            is_wiki = info[6]
            is_wiki = "" if is_wiki is None else is_wiki
            if len(is_wiki) > 0 and is_wiki in featureEncodeHashInfo["wiki"][
                "hash_info"]:
                is_wiki = featureEncodeHashInfo["wiki"]["hash_info"][is_wiki]
            else:
                is_wiki = 0

            obj_type = info[7]
            obj_type = "" if obj_type is None else obj_type
            if len(obj_type) > 0 and obj_type in featureEncodeHashInfo["obj_type"][
                "hash_info"]:
                obj_type = featureEncodeHashInfo["obj_type"]["hash_info"][obj_type]
            else:
                obj_type = 0

            topic_id = info[8]
            topic_id = 0 if topic_id is None else str(topic_id)
            if topic_id in featureEncodeHashInfo["topic_id"]["hash_info"]:
                topic_id = featureEncodeHashInfo["topic_id"]["hash_info"][topic_id]
            else:
                topic_id = 0

            topic_cate = info[9]
            topic_cate = "" if topic_cate is None else topic_cate
            if len(topic_cate) > 0 and topic_cate in featureEncodeHashInfo["topic_cate"][
                "hash_info"]:
                topic_cate = featureEncodeHashInfo["topic_cate"]["hash_info"][topic_cate]
            else:
                topic_cate = 0

            cate = info[10]
            cate = "" if cate is None else cate
            if len(cate) > 0 and cate in featureEncodeHashInfo["cate"][
                "hash_info"]:
                cate = featureEncodeHashInfo["cate"]["hash_info"][cate]
            else:
                cate = 0

            subcate = info[11]
            subcate = "" if subcate is None else subcate
            if len(subcate) > 0 and subcate in featureEncodeHashInfo["subcate"][
                "hash_info"]:
                subcate = featureEncodeHashInfo["subcate"]["hash_info"][subcate]
            else:
                subcate = 0

            author_id = info[12]
            author_id = 0 if author_id is None else str(author_id)
            if author_id in featureEncodeHashInfo["uid"]["hash_info"]:
                author_id = featureEncodeHashInfo["uid"]["hash_info"][author_id]
            else:
                author_id = 0

            author_type = info[13]
            author_type = "" if author_type is None else author_type
            if len(author_type) > 0 and author_type in featureEncodeHashInfo["user_identity_type"][
                "hash_info"]:
                author_type = featureEncodeHashInfo["user_identity_type"]["hash_info"][author_type]
            else:
                author_type = 0

            author_city = info[14]
            author_city = "" if author_city is None else author_city
            if len(author_city) > 0 and author_city in featureEncodeHashInfo["user_city"][
                "hash_info"]:
                author_city = featureEncodeHashInfo["user_city"]["hash_info"][author_city]
            else:
                author_city = 0

            gender = info[15]
            gender = "" if gender is None else gender
            if len(gender) > 0 and gender in featureEncodeHashInfo["gender"][
                "hash_info"]:
                gender = featureEncodeHashInfo["gender"]["hash_info"][gender]
            else:
                gender = 0

            item_like_7day = info[16]
            item_like_7day = 0 if item_like_7day is None else float(item_like_7day)

            item_comment_7day = info[17]
            item_comment_7day = 0 if item_comment_7day is None else float(item_comment_7day)

            item_favorite_7day = info[18]
            item_favorite_7day = 0 if item_favorite_7day is None else float(item_favorite_7day)

            item_click_7day = info[19]
            item_click_7day = 0 if item_click_7day is None else float(item_click_7day)

            item_exp_7day = info[20]
            item_exp_7day = 0 if item_exp_7day is None else float(item_exp_7day)

            item_like_30day = info[21]
            item_like_30day = 0 if item_like_30day is None else float(item_like_30day)

            item_comment_30day = info[22]
            item_comment_30day = 0 if item_comment_30day is None else float(item_comment_30day)

            item_favorite_30day = info[23]
            item_favorite_30day = 0 if item_favorite_30day is None else float(item_favorite_30day)

            item_click_30day = info[24]
            item_click_30day = 0 if item_click_30day is None else float(item_click_30day)

            item_exp_30day = info[25]
            item_exp_30day = 0 if item_exp_30day is None else float(item_exp_30day)

            author_becomment = info[26]
            author_becomment = 0 if author_becomment is None else float(author_becomment)

            author_befollow = info[27]
            author_befollow = 0 if author_befollow is None else float(author_befollow)

            author_belike = info[28]
            author_belike = 0 if author_belike is None else float(author_belike)

            author_befavorite = info[29]
            author_befavorite = 0 if author_befavorite is None else float(author_befavorite)

            objFeature = {
                "doc_id" : objIdIndex,
                "favorite_num" : favorite_num,
                "like_num" : like_num,
                "comment_num" : comment_num,
                "score" : float(score_level),
                "interval_days" : interval_week,
                "wiki" : is_wiki,
                "obj_type" : obj_type,
                "position" : 0,
                "topic_id" : topic_id,
                "topic_cate" : topic_cate,
                "cate" : cate,
                "subcate" : subcate,
                "author_uid_for_rerank" : str(author_id),
                "author_type" : author_type,
                "author_city" : author_city,
                "author_gender" : gender,
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

            hashObjFeature[objId] = objFeature


        presto_cursor.close()

        with open(fineRowHashAllDataJsonFile, 'w') as f:
            fineRowHashDataJson = {
                "hash_uid_feature": hashUserFeature,
                "hash_obj_id_feature": hashObjFeature,
            }

            f.write(json.dumps(fineRowHashDataJson))


def build_all_feature():
    buildAllFeature = BuildAllFeature()

    buildAllFeature.build_all_feature(day=1)

    log.logger.info("程序执行完毕")
    # send_msg("pkl文件数据 用户个性化重排 训练文件 生成完成")


if __name__ == '__main__':
    build_all_feature()
