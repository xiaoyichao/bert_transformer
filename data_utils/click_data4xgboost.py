# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:08:41
Description: 获取点击数据并且处理，用于训练粗排（xgboost）模型的class
conda activate py38
nohup python -u click_data4xgboost.py > click_data4xgboost.out 2>&1 &
tail -f click_data4xgboost.out
'''

import os
import sys
import random
import time
from datetime import date, datetime, timedelta
import pickle
import configparser

os.chdir(sys.path[0])
sys.path.append("../")
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from presto_con.topresto import python_presto
from common.common_log import Logger, get_log_name
from service.Tool.JiebaHHZ import JiebaHHZ
from common.tool import Tool
from tqdm import tqdm
from service.Weight.calc_weight import CalcWeight


current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

all_log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

JiebaHHZ.loadUserDict()
JiebaHHZ.del_jieba_default_words()


def get_tf(query, sentence, have_jiebad=False, max_len=64, origin_query=None):
    '''
    Author: xiaoyichao
    param {*}
    Description: 获取query 在doc中的词频
    '''
    tf_num = 0

    if have_jiebad:
        sentence_word_list = sentence[:max_len] 
        sentence = "".join(sentence)
        sentence = sentence[:max_len] 

        query_word_list = query
    else:
        query_word_list = JiebaHHZ.SplitWords(query)
        sentence_word_list = JiebaHHZ.SplitWords(sentence[:max_len], isGetUnique=False)

    for query_word in query_word_list:
        for sentence_word in sentence_word_list:
            if sentence_word == query_word:
                tf_num += 1

    if origin_query is not None:
        if origin_query in sentence:  # 给 绿色-沙发 的 数据加分，防止“绿色的植物，沙发也很软”这样query的两个词分家的数据得到高分。
            tf_num += 1
    else:
        if query in sentence:  # 给 绿色-沙发 的 数据加分，防止“绿色的植物，沙发也很软”这样query的两个词分家的数据得到高分。
            tf_num += 1

    return tf_num


class ClickData4Xgboost(object):
    '''
    Author: xiaoyichao
    param {*} self
    Description: 获取点击数据并且处理，用于训练粗排（xgboost）模型

    '''

    def __init__(self):
        self.presto = python_presto()
        self.host = root_config["prosto"]['host']
        self.port = int(root_config["prosto"]['port'])
        self.user = root_config["prosto"]['user']
        self.password = root_config["prosto"]['password']
        self.corpus = {}

    def get_click_datas(self, before_day, end_day):
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
        all_log.logger.info("presto连接完成")
        sql = '''select
                        query as "搜索词",
                        obj_id as "内容id",
                        max(relevance) as "最大相关性得分",
                        favorite_num as "内容的总收藏数",
                        like_num as "内容的总点赞数",
                        comment_num as "内容的总评论数",
                        score as "内容的后台质量分数",
                        wiki as "是否挂了wiki",
                        user_identity_type as "发布内容的用户身份",
                        obj_type as "内容类型",
                        add_day as "内容的首次发布时间",
                        day as "搜索行为发生的日期",
                        max(is_valid) as "是否是有效点击"
                    from oss.hhz.dws_search_order_model_v2
                    where  day between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null 
                    group by day,query,obj_id,favorite_num,like_num,comment_num,score,add_day,wiki,user_identity_type,obj_type,total_score,designer_intention_group_tag
                    
                    ''' % (before_day, end_day)

        all_log.logger.info("presto_cursor: " + sql)
        presto_cursor = presto_con.cursor()
        all_log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        all_log.logger.info("presto 执行 完成")
        all_log.logger.info("开始解析数据")

        click_datas = {}
        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        user_identity_type_dict = {"设计师用户": 1, "普通用户": 2, "未知": 3,
                                   "个人V认证用户": 4, "品牌用户": 5, "机构V认证用户": 6, "装修公司用户": 7, "未认证设计师用户": 8}
        # obj_type_dict = {"文章": 1, "单图": 2, "多图": 3, "整屋": 4, "视频": 5, "纯文字": 6}
        # obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        wiki_dict = {"是": 1, "否": 0}
        for info in tqdm(presto_cursor.fetchall()):
            query = info[0]
            doc_id = info[1]
            if len(doc_id) < 10:
                continue

            relevance = info[2]
            favorite_num = info[3]
            like_num = info[4]
            comment_num = info[5]
            score = info[6]
            wiki = info[7]
            user_identity_type = info[8]
            obj_type_num = Tool.getTypeByObjId(doc_id)
            add_day = info[10]
            day = info[11]
            is_valid = info[12]
            # day = time.strftime("%Y-%m-%d", time.localtime()).replace("-", "")
            # total_score = info[12]
            # designer_intention_group_tag = info[13]

            '''数据预处理'''
            favorite_num = None if favorite_num is None else favorite_num
            like_num = None if like_num is None else like_num
            comment_num = None if comment_num is None else comment_num
            score = None if score is None else score
            wiki = None if wiki is None else wiki_dict[wiki]
            # user_identity_type = None if user_identity_type is None else user_identity_type_dict[user_identity_type]
   
            obj_type = obj_type_num
            if obj_type == 0 or obj_type == 4:  # note分数转化，30转化为60
                if score is not None and score < 60:
                    score = 60

            if add_day and day:
                add_day = str(add_day)
                day = str(day)
                add_day = datetime(year=int(add_day[0:4]), month=int(
                    add_day[4:6]), day=int(add_day[6:8]))
                day = datetime(year=int(day[0:4]), month=int(
                    day[4:6]), day=int(day[6:8]))
                interval_days = (day - add_day).days
            else:
                interval_days = None

            if query not in click_datas:
                click_datas[query] = {
                    'query': query, 'positive_ids_rele': [], 'negative_ids_rele': []}
                # pos_neg_num_dict[query] = {"pos_num": 0, "neg_num": 0}

            # if relevance >= 1:  # 有交互行为的数据是相关性>1 的正样本
            #     click_datas[query]['positive_ids_rele'].append((doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
            #     corpus_obj_ids_set.add(doc_id)

            # 正样本
            if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                click_datas[query]['positive_ids_rele'].append(
                    (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                corpus_obj_ids_set.add(doc_id)
            else:
                # 正负样本比例对
                if relevance == 1 and is_valid == 1:  # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                    click_datas[query]['positive_ids_rele'].append(
                        (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type)
                    )
                    corpus_obj_ids_set.add(doc_id)
                elif relevance == 1 and is_valid == 0:
                    relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本
                    click_datas[query]['negative_ids_rele'].append(
                        (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki,
                         obj_type)
                    )
                    corpus_obj_ids_set.add(doc_id)

            # 负样本
            if relevance == 0:
                click_datas[query]['negative_ids_rele'].append(
                    (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                corpus_obj_ids_set.add(doc_id)

        return click_datas, corpus_obj_ids_set

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
            return [obj[i:i + basic_len4cut] for i in range(0, len(obj), basic_len4cut)]

        corpus = {}
        # self.add_obj_pass_ids = []
        if len(obj_ids_list) > 0:
            basic_len4cut_for_blank = 1000
            basic_len4cut_for_article = 1000
            basic_len4cut_for_photo = 10000
            basic_len4cut_for_guide = 1000
            # objid_content_type_dict = self.get_objid_content_type_dict(obj_ids_list)
            content_type_ids_dict = Tool.get_content_type_ids_dict(obj_ids_list)
            # {content_type:1是文章，2是note，3是整屋，4是指南}
            for content_type, obj_ids in content_type_ids_dict.items():
                if content_type == 1 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut_for_blank)
                    for obj_ids_1D in tqdm(obj_ids_2D):
                        _, _, _, _, objid_title_remark_dict = getProcessBlank(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 2 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut_for_photo)
                    for obj_ids_1D in tqdm(obj_ids_2D):
                        _, _, _, _, objid_title_remark_dict = getProcessNoteInfo(obj_ids_1D, use_presto=True)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 3 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut_for_article)
                    for obj_ids_1D in tqdm(obj_ids_2D):
                        _, _, _, _, objid_title_remark_dict = getProcessArticleInfo(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 4 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut_for_guide)
                    for obj_ids_1D in tqdm(obj_ids_2D):
                        _, _, _, _, objid_title_remark_dict = getProcessGuide(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                else:
                    all_log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
                    err_log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
        all_log.logger.info("corpus长度: " + str(len(corpus)))
        return corpus


    def get_samples_use_pos_neg_ration_with_text(self, click_datas, corpus, pos_neg_ration=4, max_len=64):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成train_samples
        '''
        train_samples = []
        corpus_obj_ids_set = set(corpus.keys())
        group_list = []
        group_data_sum = 0
        all_log.logger.info("run get_samples_use_pos_neg_ration_with_text")
        for query, click_data in tqdm(click_datas.items()):
            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:
                neg_cnt = 0
                pos_cnt = 0
                data_num = 0
                # query_list = JiebaHHZ.SplitWords(query)
                
                for pos_id_rele in click_data['positive_ids_rele']:
                    if pos_id_rele[0] in corpus_obj_ids_set:
                        relevance = pos_id_rele[1]
                        doc_id = pos_id_rele[0]
                        
                        title = corpus[doc_id]["title"]
                        remark = corpus[doc_id]["remark"]
                        favorite_num = pos_id_rele[2]
                        like_num = pos_id_rele[3]
                        comment_num = pos_id_rele[4]
                        score = pos_id_rele[5]
                        interval_days = pos_id_rele[6]
                        wiki = pos_id_rele[7]
                        obj_type = pos_id_rele[8]
                        tf_num4title = get_tf(query, title, max_len=max_len)
                        tf_num4remark = get_tf(query, remark, max_len=max_len)
                        if tf_num4remark == 0:  # 前max_len字如果没有包含query ，则视为负样本
                            relevance = 0
                        if len(title) > 0 and tf_num4title == 0:
                            relevance = 0
                        train_sample = [relevance]
                        #   {'f6': 7, 'f8': 5, 'f1': 2, 'f5': 7, 'f4': 2, 'f2': 3, 'f0': 1, 'f3': 4, 'f7': 1}
                        # {'f1': 5, 'f8': 5, 'f0': 10}
                        tmp_train_sample = [tf_num4title, tf_num4remark, favorite_num, like_num, comment_num, score, interval_days, obj_type]
                        for index, data in enumerate(tmp_train_sample):
                            if data is not None:
                                train_sample.append((index + 1, data))  # index+1 的原因是数据集的第一个维度的index为1，不是0，仅仅是个格式问题。
                        train_samples.append(train_sample)
                        data_num += 1
                        pos_cnt += 1

                for neg_id_rele in click_data['negative_ids_rele']:
                    if neg_id_rele[0] in corpus_obj_ids_set:
                        if neg_cnt < pos_cnt * pos_neg_ration:
                            neg_cnt += 1
                            relevance = neg_id_rele[1]
                            doc_id = neg_id_rele[0]

                            title = corpus[doc_id]["title"]
                            remark = corpus[doc_id]["remark"]
                            favorite_num = neg_id_rele[2]
                            like_num = neg_id_rele[3]
                            comment_num = neg_id_rele[4]
                            score = neg_id_rele[5]
                            interval_days = neg_id_rele[6]
                            wiki = neg_id_rele[7]
                            obj_type = neg_id_rele[8]
                            tf_num4title = get_tf(query, title, max_len=max_len)
                            tf_num4remark = get_tf(query, remark, max_len=max_len)

                            train_sample = [relevance]
                            tmp_train_sample = [tf_num4title, tf_num4remark, favorite_num, like_num, comment_num, score, interval_days, obj_type]
                            for index, data in enumerate(tmp_train_sample):
                                if data is not None:
                                    train_sample.append(
                                        (index + 1, data))  # index+1 的原因是数据集的第一个维度的index为1，不是0，仅仅是个格式问题。
                            train_samples.append(train_sample)
                            data_num += 1
                        else:
                            break
                if data_num > 0:
                    group_list.append(data_num)
                    group_data_sum += data_num

        print("train_samples 长度: " + str(len(train_samples)))
        print("group_data中的数据的和: " + str(group_data_sum))
        print("group_list 长度: " + str(len(group_list)))
        return train_samples, group_list

    def write_samples_pkl(self, samples, output_file):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将samples数据写入pkl文件
        '''
        output_file_path_list = output_file.split("/")
        output_file_parent_path = "/".join(output_file_path_list[:-1])
        if not os.path.exists(output_file_parent_path):
            os.makedirs(output_file_parent_path)
        with open(output_file, 'wb') as fo:  # 将数据写入pkl文件
            pickle.dump(samples, fo)
        print("数据写入%s文件" % str(output_file))

    def read_samples_pkl(self, pkl_file):
        '''
        Author: xiaoyichao
        param {*}
        Description: 从pkl文件中读取出samples数据
        '''
        with open(pkl_file, 'rb') as f:  # 读取pkl文件数据
            samples = pickle.load(f, encoding='bytes')
        return samples

    def write_samples2svmlight_file(self, samples, output_file):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将samples数据写入svmlight_file文件
        '''
        output_file_path_list = output_file.split("/")
        output_file_parent_path = "/".join(output_file_path_list[:-1])
        if not os.path.exists(output_file_parent_path):
            os.makedirs(output_file_parent_path)

        with open(output_file, 'w', encoding="utf-8") as f:  # 将数据写入pkl文件
            for sample in samples:
                for i, datas in enumerate(sample):
                    if i == 0:  # 相关性label
                        # relevence = str(datas)
                        relevence = datas
                        svm_format = "%s " % (relevence)
                        f.write(svm_format)
                    elif i == (len(sample) - 1):  # 最后一个维度的数据
                        index = datas[0]
                        data = datas[1]
                        svm_format = "%d:%s" % (index, data)
                        if date:
                            f.write(svm_format)
                    else:
                        index = datas[0]
                        data = datas[1]
                        svm_format = "%d:%s " % (index, data)
                        if date:
                            f.write(svm_format)
                f.write("\n")

        print("数据写入%s文件" % str(output_file))

    def write_group2txt(self, group_list, output_file):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将group_list数据写入文件
        '''
        output_file_path_list = output_file.split("/")
        output_file_parent_path = "/".join(output_file_path_list[:-1])
        if not os.path.exists(output_file_parent_path):
            os.makedirs(output_file_parent_path)
        with open(output_file, 'w', encoding="utf-8") as f:  # 将数据写入pkl文件
            for group_data in group_list:
                f.write(str(group_data) + "\n")
        print("数据写入%s文件" % str(output_file))

if __name__ == '__main__':
    print(get_tf(query="绿色沙发", sentence="绿色的植物，沙发也很软"))

#     clickdata4Xgboost = ClickData4Xgboost()
#     POS_NEG_RATION = 4
#     click_datas, train_corpus_obj_ids_set = clickdata4Xgboost.get_click_datas(
#         20211001, 1)
#     train_corpus = clickdata4Xgboost.get_corpus(train_corpus_obj_ids_set)
#     train_samples, group_list = clickdata4Xgboost.get_samples(
#         click_datas, train_corpus, POS_NEG_RATION)

#     current_dir = os.path.abspath(os.path.dirname(__file__))
#     train_samples_pkl_file = os.path.join(
#         current_dir, '../data/tf_record/train_samples4xgb.pkl')
#     train_group_pkl_file = os.path.join(
#         current_dir, '../data/tf_record/train_group4xgb.pkl')
#     clickdata4Xgboost.write_samples_pkl(
#         train_samples, train_samples_pkl_file)
#     print("train_samples_pkl_file数据已经写入pkl文件")
#     clickdata4Xgboost.write_samples_pkl(group_list, train_group_pkl_file)
#     print("train_group_pkl_file数据已经写入pkl文件")

#     train_samples_svm_file = os.path.join(
#         current_dir, '../data/xgboost_data/train_samples4xgb.train')
#     train_group_file = os.path.join(
#         current_dir, '../data/xgboost_data/train_group4xgb.train.group')
#     clickdata4Xgboost.write_samples2svmlight_file(
#         train_samples, train_samples_svm_file)
#     print("train_samples_svm_file数据已经写入pkl文件")
#     clickdata4Xgboost.write_group2txt(group_list, train_group_file)
#     print("write_group2txt数据已经写入pkl文件")
