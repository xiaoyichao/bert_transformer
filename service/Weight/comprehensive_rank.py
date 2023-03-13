# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-07-30 14:28:28
LastEditTime: 2021-08-10 15:11:03
Description: 综合排序
'''

import numpy as np
import os
import time
import configparser
from service.Parmas.comprehensive_rank_ranking_return import comprehensive_rank_ranking_return

from service.Init.InitService import InitService
from common.common_log import Logger, get_log_name
from db.hhzMember.Member import Member

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)


class ComprehensiveRank(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 综合排序 的类
    '''
    '''权重字典'''
    rank_weights = {
        "similars_with_information_weight": 10000,  # 余弦*信息量 的权重

        "jaccard4content_weight": 100,  # jaccard在内容上的相似度 的权重

        "jaccard4admintag_weight": 50,  # jaccard在后台标签上的相似度 的权重

        "interact_weight": 50,  # 单位交互比 的权重

        "admin_score_weight": 50,  # 后台内容质量 的权重

        "pool_score_weight4goods": 0.0,  # 设计师流量池 的权重
        "pool_score_weight4not_goods": 5000,  # 设计师流量池 的权重

        "wiki_score_weight4goods": 0,  # for 找商品意图，挂付费品牌wiki的内容的权重
        "wiki_score_weight4not_goods": 0,  # for 非找商品意图，挂付费品牌wiki的内容的权重
    }

    '''权重字典'''
    rank_weights_new = {
        "similars_with_information_weight": 100,  # 余弦*信息量 的权重

        "jaccard4content_weight": 1.2,  # jaccard在内容上的相似度 的权重

        "jaccard4content_full_match_weight": 0.3,  # jaccard在内容上的相似度 的权重 全匹配

        "jaccard4content_split_full_match_weight": 0.8,  # jaccard在内容上的相似度 的权重 切词匹配

        "jaccard4content_split_match_weight": 0.4,  # jaccard在内容上的相似度 的权重 切词匹配

        "title_lens_weight": 0.2,  # 标题长度 的权重

        "time_weight": 0.05,  # 时间 的权重

        "remark_lens_weight": 0.1,  # 描述长度 的权重

        "favorite_weight": 0.1,  # 收藏量 的权重

        "jaccard4admintag_weight": 0.2,  # jaccard在后台标签上的相似度 的权重

        "interact_weight": 0.2,  # 单位交互比 的权重

        "admin_score_weight": 0.1,  # 后台内容质量 的权重

        "pool_score_weight4goods": 0.2,  # 设计师流量池 的权重
        "pool_score_weight4not_goods": 0.2,  # 设计师流量池 的权重

        "wiki_score_weight4new": 0.2,  # for 新 内容绑定wiki权重
        "wiki_score_weight4newnot_goods": 0.2,  # for 新 内容绑定wiki权重
        "ctr_score_weight": 7 ,  # for ctr 得分
    }

    @classmethod
    def ranking_new(cls, comprehensive_rank_ranking_params):
        """
        将权重值 处理成 最终相加的 数据
        Args:
        Returns:
        """
        query_splited_list = comprehensive_rank_ranking_params.query_splited_list
        similars_with_information_np_TopK = comprehensive_rank_ranking_params.similars_with_information_np_TopK
        jaccard4content_np = comprehensive_rank_ranking_params.jaccard4content_np
        jaccard4admin_tag_np = comprehensive_rank_ranking_params.jaccard4admin_tag_np
        interact_time_np = comprehensive_rank_ranking_params.interact_time_np
        admin_score_np = comprehensive_rank_ranking_params.admin_score_np
        pool_score_np = comprehensive_rank_ranking_params.pool_score_np
        user_types_np = comprehensive_rank_ranking_params.user_types_np
        is_bind_wiki_np = comprehensive_rank_ranking_params.is_bind_wiki_np
        wiki_score_np = comprehensive_rank_ranking_params.wiki_score_np
        intent_list = comprehensive_rank_ranking_params.intent_list
        full_match_np = comprehensive_rank_ranking_params.full_match_np
        split_match_np = comprehensive_rank_ranking_params.split_match_np
        title_lens_weight_np = comprehensive_rank_ranking_params.title_lens_weight_np
        remark_lens_weight_np = comprehensive_rank_ranking_params.remark_lens_weight_np
        time_weight_np = comprehensive_rank_ranking_params.time_weight_np
        favorites_nums_np = comprehensive_rank_ranking_params.favorites_nums_np
        favorite_weight_np = comprehensive_rank_ranking_params.favorite_weight_np
        jaccard4content_wiki_name_np = comprehensive_rank_ranking_params.jaccard4content_wiki_name_np
        ctr_scores_np = comprehensive_rank_ranking_params.ctr_scores_np

        '''获取similars_with_information加权分数'''
        similars_with_information_np_format = cls.rank_weights_new["similars_with_information_weight"] * similars_with_information_np_TopK

        similars_with_information_stand = 0
        if len(similars_with_information_np_format) > 0 :
            similars_with_information_stand = np.percentile(similars_with_information_np_format, 99)

        print("similars_with_information_stand", similars_with_information_stand)
        ''''获取jaccard4content分数权重'''
        # jaccard4content_np_format = cls.rank_weights["jaccard4content_weight"] * jaccard4content_np * similars_with_information_np_format

        if len(query_splited_list) > 1:
            jaccard4content_full_match_np_format = cls.rank_weights_new[
                                                       "jaccard4content_full_match_weight"] * full_match_np * similars_with_information_stand
        else:
            jaccard4content_full_match_np_format = cls.rank_weights_new[
                                                       "jaccard4content_full_match_weight"] * full_match_np * similars_with_information_stand * 0


        split_full_match_flag = (split_match_np == 1.0)
        split_not_full_match_flag = (split_match_np != 1.0)

        jaccard4content_split_match_np_format = np.array(list(similars_with_information_np_format))
        jaccard4content_split_match_np_format[split_full_match_flag] = similars_with_information_np_format[split_full_match_flag] * cls.rank_weights_new["jaccard4content_split_full_match_weight"] * split_match_np[split_full_match_flag]
        jaccard4content_split_match_np_format[split_not_full_match_flag] = similars_with_information_np_format[split_not_full_match_flag] * cls.rank_weights_new["jaccard4content_split_match_weight"] * split_match_np[split_not_full_match_flag]

        ''''获取jaccard4后台标签分数权重'''
        jaccard4admin_tag_np_format = cls.rank_weights_new["jaccard4admintag_weight"] * jaccard4admin_tag_np * similars_with_information_stand

        ''''获取单位交互分数'''
        interact_time_np_format = cls.rank_weights_new["interact_weight"] * interact_time_np * similars_with_information_stand

        ''''获取后台质量分数权重'''
        admin_score_np_format = cls.rank_weights_new["admin_score_weight"] * admin_score_np * similars_with_information_stand

        ''''获取标题权重'''
        title_lens_np_format = cls.rank_weights_new["title_lens_weight"] * title_lens_weight_np * similars_with_information_stand

        ''''获取描述权重'''
        remark_lens_np_format = cls.rank_weights_new["remark_lens_weight"] * remark_lens_weight_np * similars_with_information_stand

        ''''获取收藏权重'''
        favorite_np_format = cls.rank_weights_new["favorite_weight"] * favorite_weight_np * similars_with_information_stand

        ''''获取时间权重'''
        # time_weight_flag = len(query_splited_list) == 1 and np.percentile(favorites_nums_np, 80) >= 200
        #
        # time_np_format = time_weight_np * 0
        # if time_weight_flag:
        #     time_np_format = cls.rank_weights[
        #                             "time_weight"] * time_weight_np * similars_with_information_np_format

        ''''获取wiki权重和设计师流量池权重'''
        designer_type_flag = (user_types_np == 2)
        not_designer_type_flag = (user_types_np != 2)

        user_types_list_tmp = list(user_types_np)
        user_types_np_tmp = np.array(user_types_list_tmp)

        user_types_np_tmp[designer_type_flag] = 1
        user_types_np_tmp[not_designer_type_flag] = 0

        d_designer_flag = (pool_score_np == 0.25)
        user_types_np_tmp[d_designer_flag] = 0

        pool_score_np_format = similars_with_information_stand * cls.rank_weights_new["pool_score_weight4goods"] * pool_score_np * user_types_np_tmp

        ''''内容绑定wiki得分'''
        bind_wiki_reset_list = []
        for user_type_item, is_bind_wiki_item in zip(list(user_types_np_tmp), list(is_bind_wiki_np)):
            if user_type_item == 1 and is_bind_wiki_item == 1:
                bind_wiki_reset_list.append(0)
            else:
                bind_wiki_reset_list.append(is_bind_wiki_item)

        bind_wiki_reset_np = np.array(bind_wiki_reset_list)

        wiki_score_np_format = similars_with_information_stand * cls.rank_weights_new[
            "wiki_score_weight4new"] * jaccard4content_wiki_name_np * bind_wiki_reset_np

        ''''ctr 得分'''
        ctr_score_np_format = similars_with_information_stand * ctr_scores_np * cls.rank_weights_new["ctr_score_weight"]
        # if len(intent_list) > 0:
        #     log.logger.debug("intent_list:%s" % (intent_list))
        #     if "商品" in intent_list or "找商品" in intent_list:
        #         pool_score_np_format = cls.rank_weights_new["pool_score_weight4goods"] * pool_score_np
        #
        #         ''''内容绑定wiki得分'''
        #         wiki_score_np_format = similars_with_information_np_format * cls.rank_weights_new[
        #             "wiki_score_weight4new"] * jaccard4content_wiki_name_np
        #     else:
        #         pool_score_np_format = similars_with_information_np_format * cls.rank_weights_new["pool_score_weight4not_goods"] * pool_score_np
        #
        #         ''''内容绑定wiki得分'''
        #         wiki_score_np_format = similars_with_information_np_format * cls.rank_weights_new[
        #             "wiki_score_weight4newnot_goods"] * jaccard4content_wiki_name_np
        #
        # else:  # 容错
        #     pool_score_np_format = cls.rank_weights_new["pool_score_weight4not_goods"] * pool_score_np
        #
        #     ''''内容绑定wiki得分'''
        #     wiki_score_np_format = similars_with_information_np_format * cls.rank_weights_new[
        #         "wiki_score_weight4newnot_goods"] * jaccard4content_wiki_name_np


        '''计算最终综合排序的最终得分'''
        rank_scores_with_wiki_pool_np = similars_with_information_np_format + jaccard4content_split_match_np_format + jaccard4admin_tag_np_format + interact_time_np_format + admin_score_np_format + \
                         pool_score_np_format + wiki_score_np_format + title_lens_np_format + remark_lens_np_format + favorite_np_format + ctr_score_np_format

        rank_scores_np = similars_with_information_np_format + jaccard4content_split_match_np_format + jaccard4admin_tag_np_format + interact_time_np_format + admin_score_np_format + \
                        title_lens_np_format + remark_lens_np_format + favorite_np_format + ctr_score_np_format

        # if time_weight_flag:
        #     rank_scores_np = rank_scores_np + time_np_format

        # 搜索词中是否包含非标签词
        query_splited_list_len = len(query_splited_list)
        if query_splited_list_len > 1 and len(set(query_splited_list) & InitService.community_tag) < query_splited_list_len:
            rank_scores_np = rank_scores_np + jaccard4content_full_match_np_format
        else:
            jaccard4content_full_match_np_format[jaccard4content_full_match_np_format > 0] = 0

        # rank_scores_np = similars_with_information_np_format + jaccard4content_np_format + jaccard4admin_tag_np_format + interact_time_np_format + admin_score_np_format + pool_score_np_format + wiki_score_np_format
        # rank_scores_np = similars_with_information_np_format + jaccard4content_full_match_np_format + jaccard4content_split_match_np_format + jaccard4admin_tag_np_format + interact_time_np_format + admin_score_np_format + \
        #                  pool_score_np_format + wiki_score_np_format + title_lens_np_format + remark_lens_np_format

        ranking_return = comprehensive_rank_ranking_return()
        ranking_return.rank_scores_np = rank_scores_np
        ranking_return.rank_scores_with_wiki_pool_np = rank_scores_with_wiki_pool_np
        ranking_return.similars_with_information_np_format = similars_with_information_np_format
        ranking_return.jaccard4admin_tag_np_format = jaccard4admin_tag_np_format
        ranking_return.interact_time_np_format = interact_time_np_format
        ranking_return.admin_score_np_format = admin_score_np_format
        ranking_return.pool_score_np_format = pool_score_np_format
        ranking_return.wiki_score_np_format = wiki_score_np_format
        ranking_return.jaccard4content_full_match_np_format = jaccard4content_full_match_np_format
        ranking_return.jaccard4content_split_match_np_format = jaccard4content_split_match_np_format
        ranking_return.title_lens_np_format = title_lens_np_format
        ranking_return.remark_lens_np_format = remark_lens_np_format
        ranking_return.favorite_np_format = favorite_np_format
        ranking_return.ctr_score_np_format = ctr_score_np_format

        return ranking_return

    @classmethod
    def ranking(cls, comprehensive_rank_ranking_params):
        """
        将权重值 处理成 最终相加的 数据
        Args:
        Returns:
        """
        similars_with_information_np_TopK = comprehensive_rank_ranking_params.similars_with_information_np_TopK
        jaccard4content_np = comprehensive_rank_ranking_params.jaccard4content_np
        jaccard4admin_tag_np = comprehensive_rank_ranking_params.jaccard4admin_tag_np
        interact_time_np = comprehensive_rank_ranking_params.interact_time_np
        admin_score_np = comprehensive_rank_ranking_params.admin_score_np
        pool_score_np = comprehensive_rank_ranking_params.pool_score_np
        wiki_score_np = comprehensive_rank_ranking_params.wiki_score_np
        intent_list = comprehensive_rank_ranking_params.intent_list
        # val ,title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type

        """
        将权重值 处理成 最终相加的 数据
        Args:
        Returns:
        """

        '''获取similars_with_information加权分数'''
        similars_with_information_np_format = cls.rank_weights["similars_with_information_weight"] * similars_with_information_np_TopK

        ''''获取jaccard4content分数权重'''
        jaccard4content_np_format = cls.rank_weights["jaccard4content_weight"] * jaccard4content_np

        ''''获取jaccard4后台标签分数权重'''
        jaccard4admin_tag_np_format = cls.rank_weights["jaccard4admintag_weight"] * jaccard4admin_tag_np

        ''''获取单位交互分数'''
        interact_time_np_format = cls.rank_weights["interact_weight"] * interact_time_np

        ''''获取后台质量分数权重'''
        admin_score_np_format = cls.rank_weights["admin_score_weight"] * admin_score_np

        ''''获取wiki权重和设计师流量池权重'''
        if len(intent_list) > 0:
            log.logger.debug("intent_list:%s" % (intent_list))
            if "商品" in intent_list or "找商品" in intent_list:
                wiki_score_np_format = cls.rank_weights[
                    "wiki_score_weight4goods"] * wiki_score_np
                pool_score_np_format = cls.rank_weights["pool_score_weight4goods"] * pool_score_np
            else:
                wiki_score_np_format = cls.rank_weights[
                    "wiki_score_weight4not_goods"] * wiki_score_np
                pool_score_np_format = cls.rank_weights["pool_score_weight4not_goods"] * pool_score_np

        else:  # 容错
            wiki_score_np_format = cls.rank_weights[
                "wiki_score_weight4not_goods"] * wiki_score_np

            pool_score_np_format = cls.rank_weights["pool_score_weight4not_goods"] * pool_score_np

        '''计算最终综合排序的最终得分'''

        rank_scores_np = similars_with_information_np_format + jaccard4content_np_format + jaccard4admin_tag_np_format + interact_time_np_format + admin_score_np_format + pool_score_np_format + wiki_score_np_format


        ranking_return = comprehensive_rank_ranking_return()
        ranking_return.rank_scores_np = rank_scores_np
        ranking_return.similars_with_information_np_format = similars_with_information_np_format
        ranking_return.jaccard4admin_tag_np_format = jaccard4admin_tag_np_format
        ranking_return.interact_time_np_format = interact_time_np_format
        ranking_return.admin_score_np_format = admin_score_np_format
        ranking_return.pool_score_np_format = pool_score_np_format
        ranking_return.wiki_score_np_format = wiki_score_np_format
        ranking_return.jaccard4content_full_match_np_format = wiki_score_np_format * 0
        ranking_return.jaccard4content_split_match_np_format = wiki_score_np_format * 0
        ranking_return.title_lens_np_format = wiki_score_np_format * 0
        ranking_return.remark_lens_np_format = wiki_score_np_format * 0
        ranking_return.favorite_np_format = wiki_score_np_format * 0

        return ranking_return
