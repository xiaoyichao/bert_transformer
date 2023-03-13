# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 14:30
@Desciption : 权重计算相关
'''

import math
import configparser
import time
import numpy as np
import os
import sys
import socket
import orjson
os.chdir(sys.path[0])
sys.path.append("../..")
from designer.designer_pool import DesignerPool
from service.Init.InitService import InitService
from common.common_log import Logger, get_log_name
from bert_regress.predict4tf_serving_cv import get_examples,convert_examples_to_features,tokenizer
from bert_regress.predict4tf_serving_user_profile_v2 import get_examples as user_profile_get_examples, \
    convert_examples_to_features as user_profile_convert_examples_to_features, \
    tokenizer as user_profile_tokenizer

from service.faiss_map.FaissMap import FaissMap
from service.RecallThread.GetCtrInfos import GetCtrInfos
from service.RecallThread.GetBertRankInfos import GetBertRankInfos
import datetime
from common.tool import Tool
from data_utils.click_data4bert import get_tf_web
import requests
from service.RecallThread.DataPrepareForUserProfile import DataPrepareForUserProfile
from queue import Queue

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

hostname = socket.gethostname()

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

designer_pool = DesignerPool()


'''单位时间交互与时效性'''
'''对doc较多(社区标签词+最近30天top2000)的内容：对最近的内容做加权，对较老的内容做减权'''
'''对doc较少的内容：只对最近的内容做加权，不对老的内容减权'''
cur_time = int(time.time())
last_year_time = cur_time-86400*(365*1.0)
a4before_last_year = 1
a4after_last_year = 1

'''单位时间交互与时效性：对交互较多的内容设置上限，防止交互多的一直排在顶部的位置。高交互内容随着时间的推移，变得越来越老，从时效性的角度考虑随着分母的变大，逐渐被降权'''
max_interact = 10000
max_interact_new = 1000
# max_interact = 2000
'''单位时间交互：对交互较少的内容也做限制，减少图片质量不高的数据出现的情况，防止用户感受不佳'''
'''有些新内容确实质量较高，但是因为生产的天数较少，所以交互不高，如果直接卡死交互的门槛，新的优秀内容会被沉底（对新内容的分发非常不友好），内容质量尽量通过后台打分来把控'''
# min_favoritesNumsNp = 0
min_favoritesNumsNp = 200


# 标题长度
max_title_len = 10

# 描述长度
min_remark_len = 20

# 最大交互数
max_interact_new = 500

'''对短title和remark的内容降权，防止query=客厅 的结果页头部出现 title和remark=客厅的数据'''
break_point_great_than = 2000  # 最大文本长度
scale_ratio = 10  # x轴缩放
intercept = 0.0  # 图像上移
a = 1  # (x/scale_ratio)/(a+abs(x/scale_ratio))+intercept 函数

# '''针对top query和社区标签词： 对短title和remark的内容降权，防止query=客厅 的结果页头部出现 title和remark=客厅的数据'''
# break_point_great_than4long_query = 200  # 最大文本长度
# scale_ratio4top_query = 10  # x轴缩放
# intercept4top_query = 0.2  # 图像上移
# a4top_query = 1  # (x/scale_ratio)/(a+abs(x/scale_ratio))+intercept 函数

# '''针对长尾query： '''
# break_point_great_than4tail_query = 20   # 最大文本长度
# scale_ratio4long_tail = 5
# intercept4long_tail = 0.2
# a4long_tail = 1
import socket
hostname = socket.gethostname()

class CalcWeight(object):
    # 设计师流量池的数据
    _, _, pool_set_sorted = designer_pool.get_designer_pool()
    reverse_pool_index_dict = {}
    reverse_pool_index_dict = {"a": 1.0,
                               "b": 0.5,
                               "c": 0.5,
                               "d": 0.25,
                               "general_user": 0.51}
    general_pool_data = "general_user"
    log.logger.info("流量池得分权重字典："+str(reverse_pool_index_dict))

    @classmethod
    def JaccardSim4content4xgb(cls, orgin_query, query_splited_list, contents_splited_list, contents_str_list):
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算query 和潜在问题的jaccard相似度
        '''
        sim_list = []
        query_set = set(query_splited_list)

        for content_splited, content_str in zip(contents_splited_list, contents_str_list):
            jaccard_coefficient = cls.jaccrad4content4xgb(
                query_set, orgin_query, content_splited, content_str)
            sim_list.append(jaccard_coefficient)

        return np.array(sim_list)
    @classmethod
    def jaccrad4content4xgb(cls, query_set, orgin_query, reference, content_str):  # reference为源句子，title为候选句子
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算两个句子的jaccard相似度
        '''
        gramsReference = reference

        orgin_match_num = 0
        if orgin_query in content_str:
            orgin_match_num = 1

        gramsReferenceSet = set(gramsReference)
        interactNum = len(query_set & gramsReferenceSet)

        jaccard_coefficient = float((interactNum + orgin_match_num) / (len(query_set) + 1))  # 交集

        return jaccard_coefficient

    @classmethod
    def JaccardSim4content(cls, orgin_query, query_splited_list, contents_splited_list, contents_str_list):
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算query 和潜在问题的jaccard相似度
        '''
        sim_list = []
        query_set = set(query_splited_list)

        # 原始搜索词  在 内容中完整出现
        full_match_list = []

        # 切词后的结果 在内容中存在的比例
        split_match_list = []

        for content_splited, content_str in zip(contents_splited_list, contents_str_list):
            jaccard_coefficient, orgin_match_num, split_jaccard_coefficient = cls.jaccrad4content(
                query_set, orgin_query, content_splited, content_str)
            sim_list.append(jaccard_coefficient)
            full_match_list.append(orgin_match_num)
            split_match_list.append(split_jaccard_coefficient)

        return np.array(sim_list), np.array(full_match_list, dtype=np.float32), np.array(split_match_list, dtype=np.float32)

    @classmethod
    def jaccrad4content(cls, query_set, orgin_query, reference, content_str):  # reference为源句子，title为候选句子
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算两个句子的jaccard相似度
        '''
        gramsReference = reference

        orgin_match_num = 0
        if orgin_query in content_str:
            orgin_match_num = 1

        gramsReferenceSet = set(gramsReference)
        interactNum = len(query_set & gramsReferenceSet)

        jaccard_coefficient = float((interactNum + orgin_match_num)/(len(query_set) + 1))  # 交集
        split_jaccard_coefficient = float((interactNum)/len(query_set))  # 交集

        return jaccard_coefficient, orgin_match_num, split_jaccard_coefficient

    @classmethod
    def JaccardSim4WikiNames(cls, orgin_query, query_splited_list, rela_wiki_names_splited_list, rela_wiki_names_str_list):
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算query 和潜在问题的jaccard相似度
        '''
        sim_list = []
        query_set = set(query_splited_list)

        # 原始搜索词  在 内容中完整出现
        full_match_list = []

        # 切词后的结果 在内容中存在的比例
        split_match_list = []

        for content_splited, content_str in zip(rela_wiki_names_splited_list, rela_wiki_names_str_list):
            jaccard_coefficient, orgin_match_num, split_jaccard_coefficient = cls.jaccrad4content(
                query_set, orgin_query, content_splited, content_str)
            sim_list.append(jaccard_coefficient)
            full_match_list.append(orgin_match_num)
            split_match_list.append(split_jaccard_coefficient)

        return np.array(sim_list), np.array(full_match_list, dtype=np.float32), np.array(split_match_list,
                                                                                         dtype=np.float32)

    @classmethod
    def JaccardSim4admin_tag(cls, orgin_query, query_splited_list, contents_list):
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算query 和潜在问题的jaccard相似度
        '''
        sim_list = []
        query_set = set(query_splited_list)

        for content_item in contents_list:
            jaccard_coefficient = cls.jaccrad4admin_tag(
                query_set, reference=content_item)
            sim_list.append(jaccard_coefficient)
        return np.array(sim_list)

    @classmethod
    def jaccrad4admin_tag(cls, query_set, reference):  # reference为源句子，title为候选句子
        '''
        @Author: xiaoyichao
        @param {type}
        @Description: 计算两个句子的jaccard相似度
        '''
        gramsReference = reference

        gramsReferenceSet = set(gramsReference)
        interactNum = len(query_set & gramsReferenceSet)

        jaccard_coefficient = float(interactNum / len(query_set))  # 交集
        return jaccard_coefficient

    @classmethod
    def get_title_length_weight(cls, length_np):
        '''
        x :
        y :
        '''
        maxTitleFlag = length_np > max_title_len
        notZeroTitleFlag = (length_np != 0.0)

        length_np[maxTitleFlag] = max_title_len
        # 2 * (x / 3) / (6 + (x / 3))
        x = (length_np[notZeroTitleFlag] - 15) / 3
        length_np[notZeroTitleFlag] = 2 * x / (6 + x) - 1.1

        return length_np

    @classmethod
    def get_remark_length_weight(cls, length_np):
        '''
        x :
        y :
        '''
        minRemarkFlag = (length_np < min_remark_len)
        length_np[minRemarkFlag] = min_remark_len

        # 2 * ((x - 30) / 3) / (6 + ((x - 30) / 3))
        x = (length_np - 30) / 3
        length_np = 2 * x / (6 + x)

        return length_np

    @classmethod
    def calc_information_amounts(cls, query, contents_lens_np):
        '''
        @Author: xiaoyichao
        @param {contents_lens_np:文本长度 np}
        @Description: 计算 信息量，防止用户的query=客厅，搜索结果页很多标题为客厅的内容排在头部
        x : 0~200
        y : 0~1
        '''
        contents_lens_np[contents_lens_np > break_point_great_than] = break_point_great_than
        # 归一化
        information_amounts_np = ((contents_lens_np / scale_ratio) / (a + abs(contents_lens_np / scale_ratio))) + intercept

        return information_amounts_np

    @classmethod
    def get_query_content_ctr(cls, queryVec, obj_ids, params_collect_cls, init_request_global_var):
        ctrStartTime = time.time()

        # 存放线程召回的数据
        recallDataQueue = Queue()
        # 存放召回的线程
        recallThreads = []

        pre_data = []
        objIdsUniqueList = list(set(obj_ids))
        queryVecList = queryVec.tolist()[0]

        for itemId in objIdsUniqueList:
            data = np.concatenate((queryVecList, InitService.item_model[itemId]))
            pre_data.append(data.tolist())

        doc_nums = len(objIdsUniqueList)

        splitNum = 1
        batch_num = doc_nums

        if doc_nums > 100:
            splitNum = 3
            batch_num = (doc_nums // splitNum) + 1

        startTime = time.time()
        batch_datas = []
        for i in range(splitNum):
            start = i * batch_num
            end = (i + 1) * batch_num

            param = {"instances": pre_data[start:end], "signature_name": "serving_default"}
            param = orjson.dumps(param)
            batch_datas.append(
                {"param" : param,
                 "objIdsUniqueList": objIdsUniqueList[start:end]
                 }
            )

        endTime = time.time()
        log.logger.info(hostname + " ctr json 数据 阶段的整体运行时间：{:.10f} s".format(
            endTime - startTime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        for batch_data_item in batch_datas:
            GetCoarseGraindRankThreadId = GetCtrInfos.TYPE
            GetCoarseGraindRankThread = GetCtrInfos(params_collect_cls, GetCoarseGraindRankThreadId,
                                                            recallDataQueue, init_request_global_var,
                                                            batch_data_item)
            GetCoarseGraindRankThread.setDaemon(True)
            GetCoarseGraindRankThread.start()
            recallThreads.append(GetCoarseGraindRankThread)

        for recallThread in recallThreads:
            recallThread.join()

        rank_interactions_dict = {}
        # 分配线程召回数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)

            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetCtrInfos.TYPE:
                    rank_interaction_dict_tmp = recallInfo["data"]
                    rank_interactions_dict = dict(rank_interactions_dict, **rank_interaction_dict_tmp)

        cosine_similars = []
        for obj_id in obj_ids:
            if obj_id in rank_interactions_dict:
                cosine_similars.append(rank_interactions_dict[obj_id])
            else:
                cosine_similars.append(0)

        # print("np.max(ctrScores)", np.max(cosine_similars))
        # print("np.mean(ctrScores)", np.mean(cosine_similars))
        # print("np.percentile(ctrScores, 50)", np.percentile(cosine_similars, 50))
        # print("np.percentile(ctrScores, 75)", np.percentile(cosine_similars, 75))

        ctrEndTime = time.time()
        log.logger.info(hostname + " ctr 整体 阶段的整体运行时间：{:.10f} s".format(
            ctrEndTime - ctrStartTime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        return cosine_similars, rank_interactions_dict


    @classmethod
    def get_interact_time_weight(cls, ab_test, query, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
        ''' 选择使用  获取交互和发布时间权重 的方法
        query: 搜索词
        favoritesNumsNp： 收藏数
        commentNumsNp: 评论数
        likeNumsNp: 点赞数
        publishTimeNp: 发布数
        '''
        interactTimeWeight, before_last_year_flag = cls.new_get_interact_time_weight4all(favoritesNumsNp, commentNumsNp,
                                                                                         likeNumsNp, publishTimeNp)

        favoritesNumsListTmp = list(favoritesNumsNp)
        timeWeight = cls.get_interact_time_weight4all_new(np.array(favoritesNumsListTmp, dtype=float), commentNumsNp, likeNumsNp, publishTimeNp)

        favoritesNumsListTmp2 = list(favoritesNumsNp)
        favoriteWeight = cls.get_favorite_weight(np.array(favoritesNumsListTmp2, dtype=float))

        return interactTimeWeight, before_last_year_flag, timeWeight, favoritesNumsNp, favoriteWeight
        # if query in InitService.top_and_community_tag:
        #     return cls.get_interact_time_weight_top_query(favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp)
        # else:
        #     return cls.get_interact_time_weight_long_tail_query(favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp)

    @classmethod
    def getAdminScoreWeight(cls, adminScoresNp):
        ''' 获取 分数 计算出权重值
        adminScoresNp： 分数
        '''
        admin_scores_list = list(adminScoresNp)
        new_admin_scores_list = []
        for admin_score in admin_scores_list:
            new_admin_scores_list.append(admin_score/100)
        return np.array(new_admin_scores_list)

    @classmethod
    def get_wiki_score_weight(cls, wiki_score_list, before_last_year_flag, jaccard_sims_np):
        '''
        @Author: xiaoyichao
        @param {wiki_score_list:wiki得分数据。[8,0,9,7]}
        @Description: 对那些内容下挂付费品牌的wiki内容的进行加权
        '''
        wiki_score = np.array(wiki_score_list)
        # print("origin_wiki_score", wiki_score)
        wiki_score[before_last_year_flag] = 0  # 对老数据的分数进行修改
        # print("before_last_year_flag", wiki_score)
        wiki_score[jaccard_sims_np < 1] = 0  # 对jaccard_sim !=1 的分数进行修改, 防止query 和加权的不对应
        # print("jaccard_sims_np", jaccard_sims_np)
        # print("jaccard_sims_np < 1", wiki_score)
        return wiki_score

    @classmethod
    def get_pool_score_weight(cls, pool_data_list, jaccard_sims_np):
        '''
        @Author: xiaoyichao
        @param {pool_data_list:流量池数据。['a', 'b', 'd', 'b', 'c', 'c']}
        @Description: 将设计师流量池的数据转化为分数，例如3分表示a池，2分表示b池，1分表示c池
        '''
        pool_scores_list = []
        for pool_data, jaccard_sim in zip(pool_data_list, jaccard_sims_np):
            if float(jaccard_sim) == 1.0:  # jaccard_sim == 1 的时候才给内容做设计师流量池的加权
                pool_scores_list.append(cls.reverse_pool_index_dict[pool_data])
            else:  # jaccard_sim != 1 的时候，内容视为普通用户的内容（大盘水平，取的general_user的水平）
                pool_scores_list.append(cls.reverse_pool_index_dict[cls.general_pool_data])
        return np.array(pool_scores_list)

    # @classmethod
    # def get_interact_time_weight_long_tail_query(cls, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
    #     ''' 获取交互 和 发布时间 计算出权重值
    #     favoritesNumsNp： 收藏数
    #     commentNumsNp: 评论数
    #     likeNumsNp: 点赞数
    #     publishTimeNp: 发布数
    #     '''
    #     interactNums = favoritesNumsNp + commentNumsNp + likeNumsNp
    #     publishTimeDayNp = (publishTimeNp - last_year_time) // 86400 // 7
    #     publishTimeDayNp[publishTimeDayNp == 0] = 0.5
    #     interac_ratio = interactNums / publishTimeDayNp
    #     weights = interac_ratio / (1 + abs(interac_ratio))

    #     return weights

    # @classmethod
    # def get_interact_time_weight_top_query(cls, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
    #     ''' 获取交互 和 发布时间 计算出权重值 top query和 社区标签词及别名 使用
    #     favoritesNumsNp： 收藏数
    #     commentNumsNp: 评论数
    #     likeNumsNp: 点赞数
    #     publishTimeNp: 发布数
    #     '''
    #     weights = np.array([-1 for _ in range(len(publishTimeNp))], dtype=float)
    #     interactNumsNp = favoritesNumsNp + commentNumsNp + likeNumsNp
    #     interactNumsNp[interactNumsNp > max_interact] = max_interact
    #     # 收藏数低于min_favoritesNumsNp 权重置成 -1 让交互数低的不排在前面，用户搜索结果页目前能看到的只有收藏的交互数量
    #     setZeroFlag = favoritesNumsNp < min_favoritesNumsNp

    #     # 对最近的内容做加权，对较老的内容做减权
    #     '''判断时间节点'''
    #     after_last_year_flag = publishTimeNp >= last_year_time
    #     before_last_year_flag = publishTimeNp < last_year_time
    #     time_length = -1 * (publishTimeNp - cur_time) // 86400

    #     '''当天的数据设为0.5 防止分母出现0'''
    #     time_set_0dot5_flag = time_length == 0
    #     time_length[time_set_0dot5_flag] = 0.5

    #     '''针对新内容，计算交互比'''
    #     interaction_ratio_after_last_year = (interactNumsNp[after_last_year_flag]) / (time_length[after_last_year_flag])
    #     '''计算归一化的交互比'''
    #     weights[after_last_year_flag] = interaction_ratio_after_last_year / (a4after_last_year + abs(interaction_ratio_after_last_year))

    #     '''针对老内容，计算交互比和归一化交互比'''
    #     interaction_ratio_before_last_year = (interactNumsNp[before_last_year_flag]) / (time_length[before_last_year_flag])
    #     weights[before_last_year_flag] = - interaction_ratio_before_last_year / (a4before_last_year+abs(interaction_ratio_before_last_year))

    #     '''针对当天内容，计算交互比和归一化交互比'''
    #     interaction_ratio_set_0dot5 = interactNumsNp[time_set_0dot5_flag] / time_length[time_set_0dot5_flag]
    #     weights[time_set_0dot5_flag] = interaction_ratio_set_0dot5 / (1 + abs(interaction_ratio_set_0dot5))
    #     '''交互数量过低的内容，直接将归一化交互比设置为最低数（-1）'''
    #     weights[setZeroFlag] = -1

    #     return weights

    @classmethod
    def new_get_interact_time_weight4all(cls, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
        ''' 获取交互 和 发布时间 计算出权重值
        favoritesNumsNp： 收藏数
        commentNumsNp: 评论数
        likeNumsNp: 点赞数
        publishTimeNp: 发布数
        x :
        y :
        '''
        weights = np.array([-1 for _ in range(len(publishTimeNp))], dtype=float)
        interactNumsNp = favoritesNumsNp + commentNumsNp + likeNumsNp
        interactNumsNp[interactNumsNp > max_interact_new] = max_interact_new
        # 收藏数低于min_favoritesNumsNp 权重置成 -1 让交互数低的不排在前面，用户搜索结果页目前能看到的只有收藏的交互数量
        setZeroFlag = favoritesNumsNp < min_favoritesNumsNp

        # 对最近的内容做加权，对较老的内容做减权
        '''判断时间节点'''
        after_last_year_flag = publishTimeNp >= last_year_time  # 新内容（一年内的内容）
        before_last_year_flag = publishTimeNp < last_year_time  # 老内容（一年前的内容）
        time_length = (cur_time - publishTimeNp) // 86400 // 7

        '''当天的数据设为0.5 防止分母出现0'''
        time_set_0dot5_flag = time_length == 0
        time_length[time_set_0dot5_flag] = 0.5

        '''针对新内容，计算交互比和归一化的交互比'''
        interaction_ratio_after_last_year = (interactNumsNp[after_last_year_flag]) / (time_length[after_last_year_flag])
        weights[after_last_year_flag] = interaction_ratio_after_last_year / (
                    a4after_last_year + abs(interaction_ratio_after_last_year))

        '''针对老内容，计算交互比和归一化交互比'''
        interaction_ratio_before_last_year = (interactNumsNp[before_last_year_flag]) / (
        time_length[before_last_year_flag])
        # 错的
        # weights[before_last_year_flag] = - interaction_ratio_before_last_year / (1+abs(interaction_ratio_before_last_year))
        # 对的
        weights[before_last_year_flag] = -0.5 + (
                    interaction_ratio_before_last_year / (a4before_last_year + abs(interaction_ratio_before_last_year)))

        '''针对当天内容，计算交互比和归一化交互比'''
        interaction_ratio_set_0dot5 = interactNumsNp[time_set_0dot5_flag] / time_length[time_set_0dot5_flag]
        weights[time_set_0dot5_flag] = interaction_ratio_set_0dot5 / (1 + abs(interaction_ratio_set_0dot5))
        '''交互数量过低的内容，直接将归一化交互比设置为最低数（-1）'''
        weights[setZeroFlag] = -1

        return weights, before_last_year_flag

    @classmethod
    def get_interact_time_weight4all(cls, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
        ''' 获取交互 和 发布时间 计算出权重值
        favoritesNumsNp： 收藏数
        commentNumsNp: 评论数
        likeNumsNp: 点赞数
        publishTimeNp: 发布数
        x :
        y :
        '''
        weights = np.array([-1 for _ in range(len(publishTimeNp))], dtype=float)
        interactNumsNp = favoritesNumsNp + commentNumsNp + likeNumsNp
        interactNumsNp[interactNumsNp > max_interact] = max_interact
        # 收藏数低于min_favoritesNumsNp 权重置成 -1 让交互数低的不排在前面，用户搜索结果页目前能看到的只有收藏的交互数量
        setZeroFlag = favoritesNumsNp < min_favoritesNumsNp

        # 对最近的内容做加权，对较老的内容做减权
        '''判断时间节点'''
        after_last_year_flag = publishTimeNp >= last_year_time  # 新内容（一年内的内容）
        before_last_year_flag = publishTimeNp < last_year_time  # 老内容（一年前的内容）
        time_length = (cur_time - publishTimeNp) // 86400 // 7

        '''当天的数据设为0.5 防止分母出现0'''
        time_set_0dot5_flag = time_length == 0
        time_length[time_set_0dot5_flag] = 0.5

        '''针对新内容，计算交互比和归一化的交互比'''
        interaction_ratio_after_last_year = (interactNumsNp[after_last_year_flag]) / (time_length[after_last_year_flag])
        weights[after_last_year_flag] = interaction_ratio_after_last_year / (a4after_last_year + abs(interaction_ratio_after_last_year))

        '''针对老内容，计算交互比和归一化交互比'''
        interaction_ratio_before_last_year = (interactNumsNp[before_last_year_flag]) / (time_length[before_last_year_flag])
        # 错的
        # weights[before_last_year_flag] = - interaction_ratio_before_last_year / (1+abs(interaction_ratio_before_last_year))
        # 对的
        weights[before_last_year_flag] = -1 + (interaction_ratio_before_last_year / (a4before_last_year+abs(interaction_ratio_before_last_year)))

        '''针对当天内容，计算交互比和归一化交互比'''
        interaction_ratio_set_0dot5 = interactNumsNp[time_set_0dot5_flag] / time_length[time_set_0dot5_flag]
        weights[time_set_0dot5_flag] = interaction_ratio_set_0dot5 / (1 + abs(interaction_ratio_set_0dot5))
        '''交互数量过低的内容，直接将归一化交互比设置为最低数（-1）'''
        weights[setZeroFlag] = -1

        return weights, before_last_year_flag

    @classmethod
    def get_favorite_weight(cls, favoritesNumsNp):
        # ((x-1)/20)/(0.4 + ((x-1)/20))
        favoritesNumsNp[favoritesNumsNp > 200] = 200
        favoritesNumsNp[favoritesNumsNp < 10] = 10
        data = []
        for item in favoritesNumsNp:
            calc = (math.log(item/20)) / 3
            data.append(calc)

        return np.array(data, dtype=np.float)

    @classmethod
    def get_interact_time_weight4all_new(cls, favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp):
        ''' 获取交互 和 发布时间 计算出权重值
        favoritesNumsNp： 收藏数
        commentNumsNp: 评论数
        likeNumsNp: 点赞数
        publishTimeNp: 发布数
        x :
        y :
        '''
        weights = np.array([-1 for _ in range(len(publishTimeNp))], dtype=float)

        interactNumsNp = favoritesNumsNp + commentNumsNp + likeNumsNp
        interactNumsNp[interactNumsNp > max_interact_new] = max_interact_new

        favoritesNumsNp[favoritesNumsNp > min_favoritesNumsNp] = min_favoritesNumsNp

        time_length = (cur_time - publishTimeNp) // 86400
        time_length[time_length < 1] = 1
        time_length[time_length > 900] = 900
        time_length_np = np.array(time_length, dtype=float)

        #  - (((x - 3400)/240) / (10 + ((x - 3400) / 240))) + 7.8
        x = ((time_length_np - 3400) / 240)
        time_length_np =  - (x / (10 + x)) + 7.8
        # print("time_length_np")
        # print(time_length_np)
        # print("time_length_np")
        return time_length_np

    @classmethod
    def cosine_sim_hhz(cls, x, y):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 单纯的计算两个向量的余弦相似度
        '''        
        # log.logger.info("cosine_sim_hhz函数 开始运行")
        # log.logger.info("转置函数 开始运行")
        y_T = y.T
        '''num = x.dot(y.T)'''
        # log.logger.info("转置函数 开始运行")
        # log.logger.info("点积计算 开始运行")

        # dot = 0
        # for a, b in zip(x, y_T):
        #     dot += a*b
        '''这样计算dot比上边快一些'''
        sum_list = np.array(x) * np.array(y_T)
        dot = np.sum(sum_list, axis=0)

        # log.logger.info("两个向量模长的乘积 开始运行")
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        # log.logger.info("点积/（两个向量模长的乘积） 开始计算")
        cosine_sim = dot / denom
        return np.array(cosine_sim)

    @classmethod
    def cosine_similarity_hhz(cls, X, Y):
        """
        :param x: m x k array
        :param y: n x k array
        :return: m x n array
        """
        # log.logger.debug("cosine_similarity_hhz 开始计算")
        cosine_sims_2D = []
        for x in X:
            cosine_sims_1D = []
            for y in Y:
                cosine_sim = cls.cosine_sim_hhz(x, y)
                cosine_sims_1D.append(cosine_sim)
            cosine_sims_2D.append(np.array(cosine_sims_1D))
        # log.logger.debug("cosine_similarity_hhz 计算完毕，返回结果")
        return np.array(cosine_sims_2D)

    @classmethod
    def calcMeanQuerySimilar(cls, queryContentSimilars, allMergeFaissMapIdsList):
        """
        对于同一个内容id存在多个 passage id 计算语义相似度平均值
        Args:
            queryContentSimilars:  语义相似度
            allMergeFaissMapIdsList: faiss map id
        Returns:
        """
        objIds = FaissMap.getObjIdsByIds(allMergeFaissMapIdsList, True, True)

        hashObjScore = {}
        for queryContentSimilar, objId in zip(queryContentSimilars, objIds):
            if objId in hashObjScore:
                hashObjScore[objId].append(queryContentSimilar)
            else:
                hashObjScore[objId] = [queryContentSimilar]

        returnData = []
        for objId in objIds:
            meanScore = sum(hashObjScore[objId]) / len(hashObjScore[objId])
            returnData.append(meanScore)

        return np.array(returnData)

    @classmethod
    def fclsHot(cls, favoritesNums, commentNums, likeNums, adminScores):
        '''
        param {*}
        Description: 对收藏 评论 点赞 分享 进行合并计算
        '''

        hotrankList = [math.log(favoritesNum + commentNum + likeNum + 1) * 10
                       for favoritesNum, commentNum, likeNum in zip(favoritesNums, commentNums, likeNums)]

        hotrankListNp = np.array(hotrankList)
        return hotrankListNp / 3

    @classmethod
    def getPublishTimeWeight(cls, publishTime):
        '''
        publishTime：list 发布时间戳
        Description: 对发布时间戳进行处理获得 0~1 的概率
        '''
        publishTimeNp = np.array(publishTime)
        publishTimeNp = (publishTimeNp - 1446307200) / 172800
        return publishTimeNp / 20

    @classmethod
    def getEsScore(cls, allMergeFaissMapIdsList, hashEsDataInfos):
        # 通过 FaissMapId 获取 内容id
        objIds = FaissMap.getObjIdsByIds(allMergeFaissMapIdsList, True, True)

        _scores = []

        for objId in objIds:
            if objId and objId in hashEsDataInfos:
                if "_score" in hashEsDataInfos[objId]:
                    _scores.append(hashEsDataInfos[objId]["_score"])
                else:
                    _scores.append(0)
            else:
                _scores.append(0)

        maxNum = np.max(_scores)
        if maxNum == 0:
            maxNum = 1

        return np.array(_scores) / maxNum

    @classmethod
    def topK_opt(cls, query, params_collect_cls, init_request_global_var, sortedObjIds, hashEsDataInfos, query_splited_list, topK_opt_num):
        '''
        Author: xiaoyichao
        param {*}
        Description: 对前topK_opt_num个内容进行优化排序
        '''
        begin_time = time.time()

        def get_delta_days(yesterday, today):
            dt2 = datetime.datetime.fromtimestamp(today)
            dt2 = dt2.replace(hour=0, minute=0, second=0, microsecond=0)
            dt1 = datetime.datetime.fromtimestamp(yesterday)
            dt1 = dt1.replace(hour=0, minute=0, second=0, microsecond=0)
            return (dt2 - dt1).days

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        wiki_dict = {1: "是", 0: "否"}
        sortedObjIds_top100 = sortedObjIds[:topK_opt_num]
        lines = []

        for obj_id in sortedObjIds_top100:
            line = []
            # title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type
            keys = hashEsDataInfos[obj_id].keys()
            if "title" in keys:
                title = hashEsDataInfos[obj_id]["title"]
            else:
                title = ""
            if "desc" in keys:
                remark = hashEsDataInfos[obj_id]["desc"]
            else:
                remark = ""
            if "favorite" in keys:
                favorite_num = float(hashEsDataInfos[obj_id]["favorite"])
            else:
                favorite_num = float(0)
            if "comment" in keys:
                comment_num = float(hashEsDataInfos[obj_id]["comment"])
            else:
                comment_num = float(0)
            if "like" in keys:
                like_num = float(hashEsDataInfos[obj_id]["like"])
            else:
                like_num = float(0)
            if "admin_score" in keys:
                score = float(hashEsDataInfos[obj_id]["admin_score"])
            else:
                score = float(0)
            if "publish_time" in keys:
                publish_time = int(hashEsDataInfos[obj_id]["publish_time"])
            else:
                publish_time = int(time.time())
            if "is_relate_wiki" in keys:
                wiki = hashEsDataInfos[obj_id]["is_relate_wiki"]
            else:
                wiki = 0

            obj_type_num = Tool.getTypeByObjId(obj_id)
            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict.keys() else obj_type_dict[obj_type_num]
            wiki = "否" if wiki not in wiki_dict.keys() else wiki_dict[wiki]

            if obj_type == "note" or obj_type == "回答的note":  # note分数转化，30转化为60
                if score < 60:
                    score = 60

            if "split_words_title_array" in keys:
                split_words_for_ai_title_list = hashEsDataInfos[obj_id]["split_words_title_array"]
            else:
                split_words_for_ai_title_list = []

            if "split_words_remark_array" in keys:
                split_words_for_ai_remark_list = hashEsDataInfos[obj_id]["split_words_remark_array"]
            else:
                split_words_for_ai_remark_list = []

            today = int(time.time())
            interval_days = get_delta_days(publish_time, today)
            # title_tf_num = 0
            # remark_tf_num = 0
            title_tf_num = get_tf_web(query, query_splited_list, split_words_for_ai_title_list, have_jiebad=True, max_len=64, isGetUnique = False)
            remark_tf_num = get_tf_web(query, query_splited_list, split_words_for_ai_remark_list, have_jiebad=True, max_len=64, isGetUnique = False)

            # line = ['锅盖收纳', '', '这个锅盖收纳很机智 #收纳 #厨房 #图片收集', '0.33', 0, 3, 774, 164, 12, 65, 1711, "是", "文章"]
            # line = ['迷你衣帽间', '', '迷你衣帽间,八零筑二的老公房户型,利用北向次卧凹进去的空间隔了一个小衣帽间.收纳要件由宜家艾格特系列和天马抽屉柜组成,挂衣服的区域大人的衣服分为长大衣和短大衣两个部分,以及挂小孩子短上衣的部分,推荐植绒衣架,很薄,相比一般衣架能多挂好多衣服.三平米不到,连找个拍摄角度都难,但好在一家三口的主要衣服和箱包都放进去了.', '0.33', 0, 3, 33, 9, 9, 30, 1716, "是", "note"]

            if params_collect_cls.ab_test == 41 or params_collect_cls.ab_test == 51:
                line = [query, title, remark[:(64 + 10)], '-1', title_tf_num, remark_tf_num, favorite_num, like_num,
                        comment_num, score, interval_days, wiki, obj_type, "未知用户类型", "未知设计师意向类型", 0, 1, "未知性别类型", 0, "未知房屋状态类型", 0, 0, 0, "未知装修状态类型", 0, 0, [], []]
            else:
                line = [query, title, remark[:(64 + 10)], '-1', title_tf_num, remark_tf_num, favorite_num, like_num,
                    comment_num, score, interval_days, wiki, obj_type]
            lines.append(line)
        # 多线程处理

        # 存放线程召回的数据
        recallDataQueue = Queue()

        # 存放召回的线程
        recallThreads = []

        doc_nums = len(lines)

        splitNum = 1
        batch_num = doc_nums

        if doc_nums > 200:
            splitNum = 3
            batch_num = (doc_nums // splitNum) + 1


        if params_collect_cls.ab_test == 41 or params_collect_cls.ab_test == 51:
            # 数据写入对象
            examples = user_profile_get_examples(lines)

            # 数据id化
            max_seq_length = 64

            features = user_profile_convert_examples_to_features(examples, max_seq_length, tokenizer)
        else:
            # 数据写入对象
            examples = get_examples(lines)

            # 数据id化
            max_seq_length = 64

            features = convert_examples_to_features(examples, max_seq_length, tokenizer)

        # 数据分割成多份请求
        batch_datas = []
        for i in range(splitNum):
            start = i * batch_num
            end = (i + 1) * batch_num
            param = orjson.dumps({"signature_name": "serving_default", "instances": features[start:end]})

            batch_datas.append(
                {"param": param,
                 "index": i
                 }
            )

        for batch_data_item in batch_datas:
            GetBertRankThreadId = GetBertRankInfos.TYPE
            GetBertRankThread = GetBertRankInfos(params_collect_cls, GetBertRankThreadId,
                                                            recallDataQueue, init_request_global_var,
                                                            batch_data_item)
            GetBertRankThread.setDaemon(True)
            GetBertRankThread.start()
            recallThreads.append(GetBertRankThread)

        for recallThread in recallThreads:
            recallThread.join()

        bertRankIndexs = []
        bertRankResults = []

        # 分配线程召回数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)

            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetBertRankInfos.TYPE:
                    bertRankResults.append(recallInfo["data"])
                    bertRankIndexs.append(recallInfo["index"])

        # 合并预测结果
        predictions = []
        bertRankResultIndexs = np.argsort(bertRankIndexs)
        for bertResultIndexTmp in bertRankResultIndexs:
            predictions.extend(bertRankResults[bertResultIndexTmp])

        if len(predictions) > 0:
            rank_scores_np = np.array(predictions)
            end_time = time.time()
            log.logger.info("topK opt 构造和请求数据 运行时间：{:.10f} s".format(
                end_time - begin_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            begin_time4np = time.time()
            # 获取根据分数，对索引位置排序
            sort_index_top100_index = np.argsort(-rank_scores_np)

            # 获取排序后的FaissMapId
            not_top100_sorted_obj_ids = np.array(sortedObjIds[topK_opt_num:])
            top100_sorted_obj_ids = np.array(sortedObjIds[:topK_opt_num])

            rerank_top100_sorted_obj_ids = top100_sorted_obj_ids[sort_index_top100_index]
            if (top100_sorted_obj_ids == rerank_top100_sorted_obj_ids).all():
                log.logger.info("topK opt 没 修改排序")
            else:
                log.logger.info("topK opt 修改了排序:" + str(rerank_top100_sorted_obj_ids[:5]))
            sortedObjIds = np.concatenate(
                (rerank_top100_sorted_obj_ids, not_top100_sorted_obj_ids), axis=0).tolist()
            end_time = time.time()
            log.logger.info("topK opt 请求数据后处理np数据 运行时间：{:.10f} s".format(
                end_time - begin_time4np) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            log.logger.info("topK opt 运行时间：{:.10f} s".format(
                end_time - begin_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
        else:
            log.logger.error("topK opt 失败")
            err_log.logger.error("topK opt 失败")
        return sortedObjIds

    @classmethod
    def topK_opt_user_profile(cls, query, params_collect_cls, init_request_global_var, sortedObjIds, hashEsDataInfos,
                 query_splited_list, topK_opt_num, user_profile_data = None, user_click_seq_historys = None, user_favorite_seq_historys = None):
        '''
        Author: xiaoyichao
        param {*}
        Description: 对前topK_opt_num个内容进行优化排序
        '''
        begin_time = time.time()
        sortedObjIds_top = sortedObjIds[:topK_opt_num]

        # 多线程 进行 特征预处理
        dataPrepareQueue = Queue()

        dataPrepareThreads = []
        objIdsNum = len(sortedObjIds_top)

        splitNum = 1
        batch_num = objIdsNum

        # 要进行特征处理 的 内容量 大于 66时 进行数据切分
        if objIdsNum > 66:
            splitNum = 5
            batch_num = (objIdsNum // splitNum) + 1

        start_time_prepare_data = time.time()
        # 开启多线程 进行特征处理
        for index in range(splitNum):
            start = index * batch_num
            end = (index + 1) * batch_num

            GetDataPrepareForUserProfileThread = DataPrepareForUserProfile(sortedObjIds_top[start:end], query, user_profile_data, user_click_seq_historys, user_favorite_seq_historys,
                     hashEsDataInfos, query_splited_list, params_collect_cls, init_request_global_var, dataPrepareQueue, index)
            GetDataPrepareForUserProfileThread.setDaemon(True)
            GetDataPrepareForUserProfileThread.start()
            dataPrepareThreads.append(GetDataPrepareForUserProfileThread)

        # 等待进程结束
        for dataPrepareThread in dataPrepareThreads:
            dataPrepareThread.join()

        # 分配线程 特征处理数据
        dataPrepareInfoIndexs = []
        dataPrepareInfoResults = []
        for _ in range(len(dataPrepareThreads)):
            dataPrepareInfo = dataPrepareQueue.get(block=False)

            if dataPrepareInfo is not None and type(dataPrepareInfo) == dict and 'type' in dataPrepareInfo:
                if dataPrepareInfo["type"] == DataPrepareForUserProfile.TYPE:
                    dataPrepareInfoResults.append(dataPrepareInfo["data"])
                    dataPrepareInfoIndexs.append(dataPrepareInfo["index"])

        # 合并特征结果
        features = []
        dataPrepareInfoIndexs = np.argsort(dataPrepareInfoIndexs)
        for dataPrepareInfoIndexTmp in dataPrepareInfoIndexs:
            features.extend(dataPrepareInfoResults[dataPrepareInfoIndexTmp])
        end_time_prepare_data = time.time()
        log.logger.info("topK opt 数据 预处理 运行时间：{:.10f} s".format(
            end_time_prepare_data - start_time_prepare_data) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 多线程处理
        # 存放线程召回的数据
        recallDataQueue = Queue()

        # 存放召回的线程
        recallThreads = []

        doc_nums = len(features)

        splitNum = 1
        batch_num = doc_nums

        # 数据分割成多份请求
        batch_datas = []

        for i in range(splitNum):
            start = i * batch_num
            end = (i + 1) * batch_num
            param = orjson.dumps({"signature_name": "serving_default", "instances": features[start:end]})

            batch_datas.append(
                {"param": param,
                 "index": i
                 }
            )

        for batch_data_item in batch_datas:
            GetBertRankThreadId = GetBertRankInfos.TYPE
            GetBertRankThread = GetBertRankInfos(params_collect_cls, GetBertRankThreadId,
                                                 recallDataQueue, init_request_global_var,
                                                 batch_data_item)
            GetBertRankThread.setDaemon(True)
            GetBertRankThread.start()
            recallThreads.append(GetBertRankThread)

        for recallThread in recallThreads:
            recallThread.join()

        bertRankIndexs = []
        bertRankResults = []

        # 分配线程召回数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)

            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetBertRankInfos.TYPE:
                    bertRankResults.append(recallInfo["data"])
                    bertRankIndexs.append(recallInfo["index"])

        # 合并预测结果
        predictions = []
        bertRankResultIndexs = np.argsort(bertRankIndexs)
        for bertResultIndexTmp in bertRankResultIndexs:
            predictions.extend(bertRankResults[bertResultIndexTmp])

        if len(predictions) > 0:
            rank_scores_np = np.array(predictions)
            end_time = time.time()
            log.logger.info("topK opt 构造和请求数据 运行时间：{:.10f} s".format(
                end_time - begin_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            begin_time4np = time.time()
            # 获取根据分数，对索引位置排序
            sort_index_top100_index = np.argsort(-rank_scores_np)

            # 获取排序后的FaissMapId
            not_top100_sorted_obj_ids = np.array(sortedObjIds[topK_opt_num:])
            top100_sorted_obj_ids = np.array(sortedObjIds[:topK_opt_num])

            rerank_top100_sorted_obj_ids = top100_sorted_obj_ids[sort_index_top100_index]

            # if (top100_sorted_obj_ids == rerank_top100_sorted_obj_ids).all():
            #     log.logger.info("topK opt 没 修改排序")
            # else:
            #     log.logger.info("topK opt 修改了排序:" + str(rerank_top100_sorted_obj_ids[:5]))

            sortedObjIds = np.concatenate(
                (rerank_top100_sorted_obj_ids, not_top100_sorted_obj_ids), axis=0).tolist()
            end_time = time.time()

            log.logger.info("topK opt 请求数据后处理np数据 运行时间：{:.10f} s".format(
                end_time - begin_time4np) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            log.logger.info("topK opt 运行时间：{:.10f} s".format(
                end_time - begin_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
        else:
            log.logger.error("topK opt 失败")
            err_log.logger.error("topK opt 失败")

        return sortedObjIds


if __name__ == "__main__":
    # print(CalcWeight.getAdminScoreWeight(np.array([10, 40, 50, 70, 80, 100])))

    # query_set = set(["厨房"])
    # orgin_query = "厨房"
    # content_splited = "厨房"
    # content_str = "厨房"
    # print(CalcWeight.jaccrad4content(query_set, orgin_query, content_splited, content_str))

    # wiki_data_list = [1, 1, 0, 0, 1]
    # before_last_year_flag = [True, False, True, False, False]
    # jaccard_sims_np = np.array([1.0, 0.4, 0.6, 1.0, 1.0])
    # print(CalcWeight.get_wiki_score_weight(
    #     wiki_data_list, before_last_year_flag, jaccard_sims_np))

    # pool_data_list = ['a', 'b', 'c', 'd']
    # jaccard_sims_np = [1, 0, 1, 0]
    # print(CalcWeight.get_pool_score_weight(pool_data_list, jaccard_sims_np))

    # doc_vec = np.random.random((2, 3))
    # query_vec = np.array(doc_vec[0])
    # sim = CalcWeight.cosine_sim_hhz(query_vec, query_vec)
    # print(sim)

    # doc_vec = np.random.random((3, 512))
    # query_vec = np.array([doc_vec[0]])
    # print("query_vec.shape", query_vec.shape)
    # print("doc_vec.shape", doc_vec.shape)
    # sim_list = CalcWeight.cosine_similarity_hhz(query_vec, doc_vec)
    # print(sim_list)
    # print(len(sim_list))
    # from sklearn.metrics.pairwise import cosine_similarity
    # sim_list = cosine_similarity(
    #     query_vec, doc_vec)
    # print(sim_list)
    # print(len(sim_list))

    favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp = np.array([3]), np.array([1]), np.array([4]), np.array([1618847793])
    weights, before_last_year_flag = CalcWeight.get_interact_time_weight4all(favoritesNumsNp, commentNumsNp, likeNumsNp, publishTimeNp)
    print(weights, before_last_year_flag)

