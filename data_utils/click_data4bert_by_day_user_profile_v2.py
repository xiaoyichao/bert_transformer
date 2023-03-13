# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:07:08
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
from common.common_log import Logger, get_log_name
from common.tool import Tool
from common.send_message import send_msg
from service.Tool.JiebaHHZ import JiebaHHZ
from db.hhzSearch.ObjId2Index import ObjId2Index
import numpy as np
import random


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

    if query in sentence[:128]:  # 给 绿色-沙发 的 数据加分，防止“绿色的植物，沙发也很软”这样query的两个词分家的数据得到高分。
        tf_num += 1
    return tf_num, query_word_list, sentence_word_list


def standard_scaler(data):
    '''
    Author: xiaoyichao
    param {*}
    Description: 对数值型数据进行标准化，使其数据分数的均值为0附近 ，方差为1
    '''    
    mm = StandardScaler()
    mm_data = mm.fit_transform(data)
    return mm_data


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

        # 获取 搜索词 下 用户普遍感觉好的内容 对于这些内容 不该成为负样本
        sql = '''select query, obj_id from (select count(distinct(uid)) as max_data, query, obj_id from ( select
                        query ,
                        obj_id ,
                        favorite_num ,
                        like_num ,
                        comment_num ,
                        score ,
                        add_day ,
                        wiki,
                        user_identity_type ,
                        obj_type ,
                        max(is_valid) as max_is_valid,
                        max(relevance) as max_relevance,
                        uid
                    from oss.hhz.dws_search_user_personalization_daily_report
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null and is_valid is not null and uid is not null group by query,obj_id,uid,favorite_num,like_num,comment_num,score,add_day,wiki,user_identity_type,obj_type 
                    
                    ) where max_is_valid = 1  group by query,obj_id) where max_data >= 3
                    ''' % (day)

        log.logger.info("presto_cursor: " + sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 获取 搜索下 优质内容 完成")
        log.logger.info("开始解析优质数据")

        i = 0
        hashQueryGoodContent = {}

        for info in presto_cursor.fetchall():
            query = info[0]
            query = query.strip()

            objId = info[1]

            if query in hashQueryGoodContent:
                hashQueryGoodContent[query].add(objId)
            else:
                queryListTmp = [objId]
                hashQueryGoodContent[query] = set(queryListTmp)

            if i % 1000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i = i + 1


        log.logger.info("完成解析数据")

        sql = '''select
                        query as "搜索词",
                        obj_id as "内容id",
                        favorite_num as "内容的总收藏数",
                        like_num as "内容的总点赞数",
                        comment_num as "内容的总评论数",
                        score as "内容的后台质量分数",
                        add_day as "内容的首次发布时间",
                        wiki as "是否挂了wiki",
                        user_identity_type as "发布内容的用户身份",
                        obj_type as "内容类型",
                        designer_intention_group_tag as "找设计师意向分组标签",
                        user_wiki_intention_score as "找商品意向分数",
                        position as "内容曝光位置",
                        gender as "性别",
                        age as "年龄",
                        home_status as "房屋状态",
                        home_space as "装修房屋的面积",
                        house_room_count as "户型居室",
                        budget as "装修预算",
                        decoration_status as "装修状态",
                        having_intention_for_decorate_company as "有找装修公司意愿用户",
                        user_click_history_sequence as "用户点击序列",
                        user_favorite_history_sequence as "用户收藏序列",
                        is_valid as "是否 是有效点击",
                        relevance as "相关性得分",
                        day as "搜索行为发生的日期",
                        uid as "用户uid",
                        search_request_id as "search id"
                    from oss.hhz.dws_search_user_personalization_daily_report
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null and is_valid is not null
                    ''' % (day)


        log.logger.info("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 完成")
        log.logger.info("开始解析数据")

        click_datas = {}
        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        # 所有 使用到的 内容id
        all_use_obj_id = set()

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        obj_type_dict_keys = set(obj_type_dict.keys())

        obj_type_list = ["未知内容类型", "note", "整屋", "指南", "回答的note", "文章"]
        wiki_list = ["未知wiki类型", "是", "否"]
        user_identity_type_list = ["未知用户类型", "设计师用户", "普通用户", "个人V认证用户", "品牌用户", "机构V认证用户", "装修公司用户", "未认证设计师用户"]
        designer_intention_group_tag_list = ["未知设计师意向类型", "前10%(含)", "40%～50%(含)", "30%～40%(含)", "20%～30%(含)", "60%～70%(含)", "80%～90%(含)", "50%～60%(含)", "90%～100%(含)", "10%～20%(含)", "70%～80%(含)"]
        gender_list = ["未知性别类型", "男", "女"]
        home_status_list = ["未知房屋状态类型", "毛坯房", "二手房", "精装修"]
        decoration_status_list = ["未知装修状态类型", "准备装修", "正在装修", "不需要装修"]

        i = 0

        for info in presto_cursor.fetchall():
            query = str(info[0])
            query = query.strip()

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
            obj_type_num = Tool.getTypeByObjId(doc_id)
            designer_intention_group_tag = info[10]
            user_wiki_intention_score = info[11]
            position = info[12]
            gender = info[13]
            age = info[14]
            home_status = info[15]
            home_space = info[16]
            house_room_count = info[17]
            budget = info[18]
            decoration_status = info[19]
            having_intention_for_decorate_company = info[20]
            user_click_history_sequence = info[21]
            user_favorite_history_sequence = info[22]
            is_valid = info[23]
            # day = time.strftime("%Y-%m-%d", time.localtime()).replace("-", "")

            relevance = info[24]
            day = info[25]
            uid = info[26]
            search_request_id = info[27]

            '''数据预处理'''
            favorite_num = 0 if favorite_num is None else favorite_num
            like_num = 0 if like_num is None else like_num
            comment_num = 0 if comment_num is None else comment_num

            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict_keys else obj_type_dict[obj_type_num]
            obj_type = obj_type.strip()

            if obj_type == "note" or obj_type == "回答的note":
                if score is None:
                    score = 0
                else:
                    if score < 60:
                        score = 60
            score = 0 if score is None else score

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

            wiki = "未知wiki类型" if wiki is None else wiki
            wiki = wiki.strip()

            user_identity_type = "未知用户类型" if user_identity_type is None or len(user_identity_type.strip()) == 0 or user_identity_type.strip() == "未知" else user_identity_type
            user_identity_type.strip()

            designer_intention_group_tag = "未知设计师意向类型" if designer_intention_group_tag is None else designer_intention_group_tag
            designer_intention_group_tag = designer_intention_group_tag.strip()

            user_wiki_intention_score = 0.0 if user_wiki_intention_score is None else float(user_wiki_intention_score)
            if user_wiki_intention_score < 0:
                user_wiki_intention_score = 0.0

            position = 0 if position is None else int(position)
            if position < 0:
                position = 0

            gender = "未知性别类型" if gender is None or gender.strip() == "未知" else gender
            gender = gender.strip()

            age = 0 if age is None else int(age)
            if age < 0:
                age = 0

            home_status = "未知房屋状态类型" if home_status is None or home_status.strip() == "未知" else home_status
            home_status = home_status.strip()

            home_space = 0 if home_space is None else int(home_space)

            house_room_count = 0 if house_room_count is None or not str(house_room_count).isdigit() else int(house_room_count)

            budget = 0 if budget is None or not str(budget).isdigit() else int(budget)

            decoration_status = "未知装修状态类型" if decoration_status is None or decoration_status.strip() == "未知" else decoration_status.strip()
            decoration_status = decoration_status.strip()

            having_intention_for_decorate_company = 0 if having_intention_for_decorate_company is None or not str(having_intention_for_decorate_company).isdigit() else int(having_intention_for_decorate_company)

            user_click_history_sequence = "" if user_click_history_sequence is None else user_click_history_sequence.strip()

            user_favorite_history_sequence = "" if user_favorite_history_sequence is None else user_favorite_history_sequence.strip()

            # 记录所有 使用过 的 内容id
            all_use_obj_id.add(doc_id)
            for click_item_seq_id in user_click_history_sequence.split(","):
                click_item_seq_id = click_item_seq_id.strip()
                if len(click_item_seq_id) > 10:
                    all_use_obj_id.add(click_item_seq_id)

            for favorite_item_seq_id in user_favorite_history_sequence.split(","):
                favorite_item_seq_id = favorite_item_seq_id.strip()
                if len(favorite_item_seq_id) > 10:
                    all_use_obj_id.add(favorite_item_seq_id)

            # unique_str = str(query) + "_" + str(uid) + "_" + str(search_request_id)
            unique_str = str(query)

            if unique_str not in click_datas:
                click_datas[unique_str] = {
                    'query': query, 'positive_ids_rele': [], 'negative_ids_rele': []}


            one_sample = (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type,
             user_identity_type, designer_intention_group_tag, user_wiki_intention_score, position, gender, age,
             home_status, home_space, house_room_count, budget, decoration_status,
             having_intention_for_decorate_company, user_click_history_sequence, user_favorite_history_sequence, str(search_request_id))

            # 正样本
            if relevance > 0:
                if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                    click_datas[unique_str]['positive_ids_rele'].append(one_sample)
                    corpus_obj_ids_set.add(doc_id)
                else:
                    # 正负样本比例对
                    if relevance == 1 and is_valid == 1: # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                        click_datas[unique_str]['positive_ids_rele'].append(one_sample)
                        corpus_obj_ids_set.add(doc_id)
                    else:
                        # 优质样本 不做负样本
                        if query in hashQueryGoodContent and doc_id in hashQueryGoodContent[query]:
                            continue

                        relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本

                        one_sample = list(one_sample)
                        one_sample[1] = relevance
                        one_sample = tuple(one_sample)

                        click_datas[unique_str]['negative_ids_rele'].append(one_sample)
                        corpus_obj_ids_set.add(doc_id)

            # 负样本
            else:
                # 优质样本 不做负样本
                if query in hashQueryGoodContent and doc_id in hashQueryGoodContent[query]:
                    continue

                click_datas[unique_str]['negative_ids_rele'].append(one_sample)
                corpus_obj_ids_set.add(doc_id)

            if i % 10000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i += 1

        log.logger.info("完成解析数据")
        return click_datas, corpus_obj_ids_set, all_use_obj_id

    def get_corpus(self, obj_ids_list):
        '''
        Author: xiaoyichao
        param {*} self
        param {*} obj_ids_list
        Description: 获取obj_id 和 文本 组成的字典
        '''

        def cut(obj, basic_len4cut):
            '''
            Author: xiaoyichao
            param {*}
            Description: 按照长度且切分list,比如5000个数据切一次。但是有的训练数据已经被用户删除了，所以数据开会变少。
            '''
            return [obj[i:i+basic_len4cut] for i in range(0, len(obj), basic_len4cut)]

        corpus = {}
        # self.add_obj_pass_ids = []
        if len(obj_ids_list) > 0:
            basic_len4cut = 500
            # objid_content_type_dict = self.get_objid_content_type_dict(obj_ids_list)
            content_type_ids_dict = Tool.get_content_type_ids_dict(obj_ids_list)
            # {content_type:1是文章，2是note，3是整屋，4是指南}
            for content_type, obj_ids in content_type_ids_dict.items():
                if content_type == 1 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessBlank(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 2 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, 5000)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessNoteInfo(obj_ids_1D, use_presto=True)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 3 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessArticleInfo(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 4 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessGuide(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                else:
                    log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
                    err_log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
        log.logger.info("corpus长度: "+str(len(corpus)))
        return corpus

    def get_samples(self, click_datas, corpus, max_relevance=3, pos_neg_ration=4, max_len=32):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成samples
        '''
        samples = []
        corpus_obj_ids_set = set(corpus.keys())

        # 获取 内容id 数字索引
        objId2IndexMap = getObjId2Index()

        user_click_history_sequence_max_len = 40
        user_favorite_history_sequence_max_len = 40

        positive_data_num = 0
        for unique_str, click_data in click_datas.items():
            query = click_data['query']

            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:
                neg_cnt = 0
                pos_cnt = 0
                for pos_id_rele_data in click_data['positive_ids_rele']:
                    objId = pos_id_rele_data[0]

                    if objId in corpus_obj_ids_set and objId in objId2IndexMap:
                        user_click_history_sequence = pos_id_rele_data[-3]
                        user_click_history_obj2index_sequence = []

                        if len(user_click_history_sequence) > 0:
                            user_click_history_sequence_list = user_click_history_sequence.split(",")
                            for user_click_history_sequence_item in user_click_history_sequence_list:
                                user_click_history_sequence_item = str(user_click_history_sequence_item).strip()
                                if len(
                                        user_click_history_sequence_item) > 0 and user_click_history_sequence_item in objId2IndexMap:
                                    obj2index = objId2IndexMap[user_click_history_sequence_item]
                                    user_click_history_obj2index_sequence.append(obj2index)
                        user_click_history_obj2index_sequence = user_click_history_obj2index_sequence[-user_click_history_sequence_max_len:]


                        user_favorite_history_sequence = pos_id_rele_data[-2]
                        user_favorite_history_obj2index_sequence = []
                        if len(user_favorite_history_sequence) > 0:
                            user_favorite_history_sequence_list = user_favorite_history_sequence.split(",")
                            for user_favorite_history_sequence_item in user_favorite_history_sequence_list:
                                user_favorite_history_sequence_item = str(user_favorite_history_sequence_item).strip()
                                if len(
                                        user_favorite_history_sequence_item) > 0 and user_favorite_history_sequence_item in objId2IndexMap:
                                    obj2index = objId2IndexMap[user_favorite_history_sequence_item]
                                    user_favorite_history_obj2index_sequence.append(obj2index)
                        user_favorite_history_obj2index_sequence = user_favorite_history_obj2index_sequence[-user_favorite_history_sequence_max_len:]

                        # 这个case 会把那些同义词的结果都变成负样本
                        title = corpus[objId]["title"]
                        remark = corpus[objId]["remark"]

                        title_tf_num, query_word_list, title_word_list = get_tf(query, title, have_jiebad=False, max_len=64)  # 这个max_len不需要跟bert的max_len一样
                        remark_tf_num, query_word_list, remark_word_list = get_tf(query, remark, have_jiebad=False, max_len=64)

                        query_word_set = set(query_word_list)
                        query_word_len = len(query_word_set)

                        if query in title[:max_len] or query in remark[:max_len] or \
                                (len(query_word_set & set(title_word_list)) == query_word_len) or (len(query_word_set & set(remark_word_list)) == query_word_len):
                            # 记录 总正样本数量
                            positive_data_num += 1
                            pos_cnt += 1
                            label = round(float(pos_id_rele_data[1]/max_relevance), 2)
                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type,
                            #              user_identity_type, designer_intention_group_tag, user_wiki_intention_score, position, gender, age,
                            #              home_status, home_space, house_room_count, budget, decoration_status,
                            #              having_intention_for_decorate_company, user_click_history_sequence, user_favorite_history_sequence]
                            for d in pos_id_rele_data[2:-3]:
                                data4train.append(d)

                            # 当前obj id
                            data4train.append(objId2IndexMap[objId])
                            data4train.append(user_click_history_obj2index_sequence)
                            data4train.append(user_favorite_history_obj2index_sequence)

                            samples.append(data4train)
                            # samples = [query, title, remark, str(label), title_tf_num, remark_tf_num, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type,
                            #              user_identity_type, designer_intention_group_tag, user_wiki_intention_score, position, gender, age,
                            #              home_status, home_space, house_room_count, budget, decoration_status,
                            #              having_intention_for_decorate_company, obj_id, user_click_history_sequence, user_favorite_history_sequence]

                        else:
                            # 样本正负比例
                            # if neg_cnt < pos_cnt*pos_neg_ration:
                            # 伪正样本 不记录为 负样本数
                            # neg_cnt += 1
                            label = 0
                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # neg_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type,
                            #              user_identity_type, designer_intention_group_tag, user_wiki_intention_score, position, gender, age,
                            #              home_status, home_space, house_room_count, budget, decoration_status,
                            #              having_intention_for_decorate_company, user_click_history_sequence, user_favorite_history_sequence]
                            for d in pos_id_rele_data[2:-3]:
                                data4train.append(d)

                            # 当前obj id
                            data4train.append(objId2IndexMap[objId])
                            data4train.append(user_click_history_obj2index_sequence)
                            data4train.append(user_favorite_history_obj2index_sequence)

                            samples.append(data4train)
                            # samples = [query, title, remark, str(label), title_tf_num, remark_tf_num, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type,
                            #              user_identity_type, designer_intention_group_tag, user_wiki_intention_score, position, gender, age,
                            #              home_status, home_space, house_room_count, budget, decoration_status,
                            #              having_intention_for_decorate_company, obj_id, user_click_history_sequence, user_favorite_history_sequence]

                            # else:
                            #     break

                random.shuffle(click_data['negative_ids_rele'])

                for neg_id_rele_data in click_data['negative_ids_rele']:
                    objId = neg_id_rele_data[0]

                    if objId in corpus_obj_ids_set and objId in objId2IndexMap:
                        user_click_history_sequence = neg_id_rele_data[-3]
                        user_click_history_obj2index_sequence = []

                        if len(user_click_history_sequence) > 0:
                            user_click_history_sequence_list = user_click_history_sequence.split()
                            for user_click_history_sequence_item in user_click_history_sequence_list:
                                user_click_history_sequence_item = str(user_click_history_sequence_item).strip()
                                if len(
                                        user_click_history_sequence_item) > 0 and user_click_history_sequence_item in objId2IndexMap:
                                    obj2index = objId2IndexMap[user_click_history_sequence_item]
                                    user_click_history_obj2index_sequence.append(obj2index)
                        user_click_history_obj2index_sequence = user_click_history_obj2index_sequence[-user_click_history_sequence_max_len:]

                        user_favorite_history_sequence = neg_id_rele_data[-2]
                        user_favorite_history_obj2index_sequence = []
                        if len(user_favorite_history_sequence) > 0:
                            user_favorite_history_sequence_list = user_favorite_history_sequence.split()
                            for user_favorite_history_sequence_item in user_favorite_history_sequence_list:
                                user_favorite_history_sequence_item = str(user_favorite_history_sequence_item).strip()
                                if len(
                                        user_favorite_history_sequence_item) > 0 and user_favorite_history_sequence_item in objId2IndexMap:
                                    obj2index = objId2IndexMap[user_favorite_history_sequence_item]
                                    user_favorite_history_obj2index_sequence.append(obj2index)
                        user_favorite_history_obj2index_sequence = user_favorite_history_obj2index_sequence[-user_favorite_history_sequence_max_len:]

                        if neg_cnt < pos_cnt * pos_neg_ration:
                            neg_cnt += 1

                            label = 0
                            title = corpus[objId]["title"]
                            remark = corpus[objId]["remark"]
                            title_tf_num, _, _ = get_tf(query, title, have_jiebad=False, max_len=64)  # 这个max_len不需要跟bert的max_len一样
                            remark_tf_num, _, _ = get_tf(query, remark, have_jiebad=False, max_len=64)

                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]

                            for d in neg_id_rele_data[2:-3]:
                                data4train.append(d)

                            # 当前obj id
                            data4train.append(objId2IndexMap[objId])
                            data4train.append(user_click_history_obj2index_sequence)
                            data4train.append(user_favorite_history_obj2index_sequence)

                            samples.append(data4train)
                        else:
                            break
        log.logger.info("samples 长度: "+str(len(samples)))
        log.logger.info("所有正样本的 数量: "+str(positive_data_num))
        return samples

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

from service.Init.InitResource import InitResource
# 获取 内容 id 对应 index
def getObjId2Index():
    InitResourceCls = InitResource()
    InitResourceCls.initDbSearchConnect()

    objId2Index = ObjId2Index.getAllObjId2Index(InitResourceCls)
    InitResourceCls.closeDbSearchConnect()

    return objId2Index

# 将使用过的内容id插入表中 获得唯一数字
def insertNewUseObjIds(all_use_obj_id):
    InitResourceCls = InitResource()
    InitResourceCls.initDbSearchConnect()

    objId2Index = getObjId2Index()
    all_db_objIds = set(objId2Index.keys())

    all_use_obj_ids_set = set(all_use_obj_id)
    diff_obj_ids = all_use_obj_ids_set - all_db_objIds
    if len(diff_obj_ids) > 0:
        need_insert_data = []

        for diff_obj_id in diff_obj_ids:
            need_insert_data.append(diff_obj_id)

            if len(need_insert_data) > 10:
                ObjId2Index.insertDatas(InitResourceCls, need_insert_data)
                need_insert_data = []

        if len(need_insert_data) > 0:
            ObjId2Index.insertDatas(InitResourceCls, need_insert_data)

    InitResourceCls.closeDbSearchConnect()

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
        click_datas, corpus_obj_ids_set, all_use_obj_id = clickdata.get_click_datas4bert(day=day)
        insertNewUseObjIds(all_use_obj_id)

        corpus_obj_ids_set_add = corpus_obj_ids_set - corpus_obj_ids_set_sum

        log.logger.info("len(click_datas): " + str(len(click_datas)))

        corpus = clickdata.get_corpus(corpus_obj_ids_set_add)

        corpus_sum.update(corpus)

        corpus_obj_ids_set_sum = corpus_obj_ids_set_sum | corpus_obj_ids_set
        log.logger.info("len(corpus_sum): " + str(len(corpus_sum)))
        samples = clickdata.get_samples(click_datas=click_datas, corpus=corpus_sum, max_relevance=max_relevance, pos_neg_ration=POS_NEG_RATION, max_len=max_len)
        log.logger.info("len(samples): " + str(len(samples)))

        pkl_file = os.path.join('/data/search_opt_model/topk_opt/rank_data_user_profile_v2/data_'+get_before_day(day)+'.pkl')
        write_samples_pkl(samples, pkl_file)

        log.logger.info("数据读取完成"+ str(day))
    log.logger.info("程序执行完毕")
    # send_msg("pkl文件数据 rank_data_more_1_4_tf 生成完成")


if __name__ == '__main__':
    get_click_data()
