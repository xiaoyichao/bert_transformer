# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 15:27
@Desciption : matching
'''
import numpy as np
import os
import time
import configparser
from service.data_process import DataProcess
from service.Weight.calc_weight import CalcWeight
from service.faiss_map.FaissMap import FaissMap
from common.common_log import Logger, get_log_name
from service.Parmas.matching_return import matching_return

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)


class MatchingOp(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: Matching 的类
    '''

    @classmethod
    def matching_for_xgb(self, params_collect_cls, objIds, hashEsDataInfos, query, init_request_global_var,
                 query_splited_list):
        '''
        @Author: xiaoyichao, zhongjie
        @param {*}
        @Description: 通过 FaissMapId 获取 内容的 语义相似度， jaccard相似度，时效性，等数据
        '''
        # 获取内容相关数据 用于综合排序
        get_content_info4weight_return_cls = DataProcess.get_content_infos4weight_calc4xgb(
            objIds, hashEsDataInfos)

        contents_splited_list = get_content_info4weight_return_cls.contents_splited_list

        contents_str_list = get_content_info4weight_return_cls.contents_str_list

        hash_obj_user_type = get_content_info4weight_return_cls.hash_obj_user_type
        hash_obj_related_wiki = get_content_info4weight_return_cls.hash_obj_related_wiki
        hash_main_service_area = get_content_info4weight_return_cls.hash_main_service_area
        publish_time = get_content_info4weight_return_cls.publish_time
        hash_obj_split_title_for_ai = get_content_info4weight_return_cls.hash_obj_split_title_for_ai
        hash_obj_split_remark_for_ai = get_content_info4weight_return_cls.hash_obj_split_remark_for_ai
        favorite_nums = get_content_info4weight_return_cls.favorite_nums
        like_nums = get_content_info4weight_return_cls.like_nums
        comment_nums = get_content_info4weight_return_cls.comment_nums
        admin_scores = get_content_info4weight_return_cls.admin_scores
        wiki_score_list = get_content_info4weight_return_cls.wiki_score_list

        start_time4jaccard_content = time.time()
        jaccard4content_np = CalcWeight.JaccardSim4content4xgb(query, query_splited_list,
                                                                                          contents_splited_list,
                                                                                          contents_str_list)
        end_time4jaccard_content = time.time()
        used_time4jaccard_content = end_time4jaccard_content - start_time4jaccard_content
        log.logger.info("计算 文本内容 的Jaccard Sim 运行时间：{:.10f} s".format(
            used_time4jaccard_content) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 获取内容id 与 Jaccard Sim 的值
        hash_obj_id_jaccard_sim = {}

        for obj_id, jaccard_sim in zip(objIds, jaccard4content_np):
            if len(obj_id) > 0:
                hash_obj_id_jaccard_sim[obj_id] = jaccard_sim

        matching_return_cls = matching_return()

        matching_return_cls.hash_obj_user_type = hash_obj_user_type
        matching_return_cls.hash_obj_id_jaccard_sim = hash_obj_id_jaccard_sim
        matching_return_cls.hash_obj_related_wiki = hash_obj_related_wiki
        matching_return_cls.hash_main_service_area = hash_main_service_area
        matching_return_cls.publish_time = publish_time
        matching_return_cls.hash_obj_split_title_for_ai = hash_obj_split_title_for_ai
        matching_return_cls.hash_obj_split_remark_for_ai = hash_obj_split_remark_for_ai
        matching_return_cls.favorite_nums = favorite_nums
        matching_return_cls.like_nums = like_nums
        matching_return_cls.comment_nums = comment_nums
        matching_return_cls.admin_scores = admin_scores
        matching_return_cls.wiki_score_list = wiki_score_list

        return matching_return_cls


    @classmethod
    def matching(self, params_collect_cls, allMergeFaissMapIdsList, hashEsDataInfos, query, init_request_global_var, query_splited_list):
        '''
        @Author: xiaoyichao, zhongjie
        @param {*}
        @Description: 通过 FaissMapId 获取 内容的 语义相似度， jaccard相似度，时效性，等数据
        '''
        objIds = FaissMap.getObjIdsByIds(allMergeFaissMapIdsList, True, True)

        # 获取内容相关数据 用于综合排序
        get_content_info4weight_return_cls = DataProcess.get_content_infos4weight_calc(
                objIds, query, hashEsDataInfos, params_collect_cls)

        wiki_score_list = get_content_info4weight_return_cls.wiki_score_list
        pool_data_list = get_content_info4weight_return_cls.pool_data_list
        likeNums = get_content_info4weight_return_cls.like_nums
        favoriteNums = get_content_info4weight_return_cls.favorite_nums
        commentNums = get_content_info4weight_return_cls.comment_nums
        publishTime = get_content_info4weight_return_cls.publish_time
        adminScores = get_content_info4weight_return_cls.admin_scores
        contents_splited_list = get_content_info4weight_return_cls.contents_splited_list
        contentAdminTags = get_content_info4weight_return_cls.content_admin_tags
        contents_lens = get_content_info4weight_return_cls.contents_lens
        contents_str_list = get_content_info4weight_return_cls.contents_str_list
        user_types = get_content_info4weight_return_cls.user_types
        hash_obj_user_type = get_content_info4weight_return_cls.hash_obj_user_type
        hash_obj_related_wiki = get_content_info4weight_return_cls.hash_obj_related_wiki
        title_lens = get_content_info4weight_return_cls.title_lens
        remark_lens = get_content_info4weight_return_cls.remark_lens
        rela_wiki_names_splited_list = get_content_info4weight_return_cls.rela_wiki_names_splited_list
        rela_wiki_names_str_list = get_content_info4weight_return_cls.rela_wiki_names_str_list
        is_bind_wiki_list = get_content_info4weight_return_cls.is_bind_wiki_list
        hash_obj_designer_flow = get_content_info4weight_return_cls.hash_obj_designer_flow
        hash_main_service_area = get_content_info4weight_return_cls.hash_main_service_area

        start_time = time.time()
        interact_time_np, before_last_year_flag, timeWeight, favoritesNumsNp, favoriteWeightNp = CalcWeight.get_interact_time_weight(params_collect_cls.ab_test, query, np.array(
            favoriteNums, dtype="float32"), np.array(commentNums, dtype="float32"), np.array(likeNums, dtype="float32"), np.array(publishTime, dtype="float32"))
        end_time = time.time()
        # log.logger.debug("before_last_year_flag"+str(before_last_year_flag))
        log.logger.info("计算 单位交互比 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 内容绑定wiki相关性
        start_time4jaccard_wiki = time.time()
        jaccard4content_wiki_name_np, full_match_wiki_name_np, split_match_wiki_name_np = CalcWeight.JaccardSim4WikiNames(query, query_splited_list,
                                                                                          rela_wiki_names_splited_list,
                                                                                          rela_wiki_names_str_list)
        end_time4jaccard_wiki = time.time()
        used_time4jaccard_wiki = end_time4jaccard_wiki - start_time4jaccard_wiki
        log.logger.info("计算 wiki 标题 的Jaccard Sim 运行时间：{:.10f} s".format(
            used_time4jaccard_wiki) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)


        start_time4jaccard_content = time.time()
        jaccard4content_np, full_match_np, split_match_np = CalcWeight.JaccardSim4content(query, query_splited_list, contents_splited_list, contents_str_list)
        end_time4jaccard_content = time.time()
        used_time4jaccard_content = end_time4jaccard_content - start_time4jaccard_content
        log.logger.info("计算 文本内容 的Jaccard Sim 运行时间：{:.10f} s".format(
            used_time4jaccard_content) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 获取内容id 与 Jaccard Sim 的值
        hash_obj_id_jaccard_sim = {}
        hash_obj_id_jaccard_split_match_sim = {}

        for obj_id, jaccard_sim, split_match in zip(objIds, jaccard4content_np, split_match_np):
            if len(obj_id) > 0:
                hash_obj_id_jaccard_sim[obj_id] = jaccard_sim
                hash_obj_id_jaccard_split_match_sim[obj_id] = split_match

        start_time4wiki = time.time()
        wiki_score_np = CalcWeight.get_wiki_score_weight(wiki_score_list, before_last_year_flag, jaccard4content_np)
        end_time4wiki = time.time()
        used_time4wiki = end_time4wiki - start_time4wiki
        log.logger.info("计算 wiki分数 运行时间：{:.10f} s".format(
            used_time4wiki) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time4pool = time.time()
        pool_score_np = CalcWeight.get_pool_score_weight(pool_data_list, jaccard4content_np)
        end_time4pool = time.time()
        used_time4pool = end_time4pool - start_time4pool
        log.logger.info("计算 流量池分数 运行时间：{:.10f} s".format(
            used_time4pool) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        admin_score_np = CalcWeight.getAdminScoreWeight(
            np.array(adminScores, "float32"))
        end_time = time.time()
        log.logger.info("计算 后台质量分数 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        jaccard4admin_tag_np = CalcWeight.JaccardSim4admin_tag(query, query_splited_list, contentAdminTags)
        end_time = time.time()
        log.logger.info("计算 后台标签 的Jaccard Sim 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        title_lens_weight_np = CalcWeight.get_title_length_weight(np.array(title_lens, dtype=np.float))
        end_time = time.time()
        log.logger.info("计算 文本标题长度 的权重 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        remark_lens_weight_np = CalcWeight.get_remark_length_weight(np.array(remark_lens, dtype=np.float))
        # print("remark_lens_weight_np", remark_lens_weight_np)
        end_time = time.time()
        log.logger.info("计算 文本描述长度 的权重 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        matching_return_cls = matching_return()
        matching_return_cls.jaccard4content_np = jaccard4content_np
        matching_return_cls.jaccard4admin_tag_np = jaccard4admin_tag_np
        matching_return_cls.interact_time_np = interact_time_np
        matching_return_cls.admin_score_np = admin_score_np
        matching_return_cls.pool_score_np = pool_score_np
        matching_return_cls.wiki_score_np = wiki_score_np
        matching_return_cls.user_types_np = np.array(user_types)
        matching_return_cls.hash_obj_user_type = hash_obj_user_type
        matching_return_cls.hash_obj_id_jaccard_sim = hash_obj_id_jaccard_sim
        matching_return_cls.hash_obj_related_wiki = hash_obj_related_wiki
        matching_return_cls.full_match_np = full_match_np
        matching_return_cls.split_match_np = split_match_np
        matching_return_cls.title_lens_weight_np = title_lens_weight_np
        matching_return_cls.remark_lens_weight_np = remark_lens_weight_np
        matching_return_cls.time_weight_np = timeWeight
        matching_return_cls.favorites_nums_np = favoritesNumsNp
        matching_return_cls.pool_data_list = pool_data_list
        matching_return_cls.jaccard4content_wiki_name_np = jaccard4content_wiki_name_np
        matching_return_cls.hash_obj_id_jaccard_split_match_sim = hash_obj_id_jaccard_split_match_sim
        matching_return_cls.is_bind_wiki_np = np.array(is_bind_wiki_list)
        matching_return_cls.favorite_weight_np = favoriteWeightNp
        matching_return_cls.hash_obj_designer_flow = hash_obj_designer_flow
        matching_return_cls.hash_main_service_area = hash_main_service_area

        return  matching_return_cls

    @classmethod
    def matching_new(self, params_collect_cls, esRecallObjIdsList_TopK, hashEsDataInfos, query, init_request_global_var,
                 query_splited_list):
        '''
        @Author: xiaoyichao, zhongjie
        @param {*}
        @Description: 通过 FaissMapId 获取 内容的 语义相似度， jaccard相似度，时效性，等数据
        '''
        objIds = esRecallObjIdsList_TopK

        # 获取内容相关数据 用于综合排序
        get_content_info4weight_return_cls = DataProcess.get_content_infos4weight_calc(
            objIds, query, hashEsDataInfos, params_collect_cls)

        wiki_score_list = get_content_info4weight_return_cls.wiki_score_list
        pool_data_list = get_content_info4weight_return_cls.pool_data_list
        likeNums = get_content_info4weight_return_cls.like_nums
        favoriteNums = get_content_info4weight_return_cls.favorite_nums
        commentNums = get_content_info4weight_return_cls.comment_nums
        publishTime = get_content_info4weight_return_cls.publish_time
        adminScores = get_content_info4weight_return_cls.admin_scores
        contents_splited_list = get_content_info4weight_return_cls.contents_splited_list
        contentAdminTags = get_content_info4weight_return_cls.content_admin_tags
        contents_lens = get_content_info4weight_return_cls.contents_lens
        contents_str_list = get_content_info4weight_return_cls.contents_str_list
        user_types = get_content_info4weight_return_cls.user_types
        hash_obj_user_type = get_content_info4weight_return_cls.hash_obj_user_type
        hash_obj_related_wiki = get_content_info4weight_return_cls.hash_obj_related_wiki
        title_lens = get_content_info4weight_return_cls.title_lens
        remark_lens = get_content_info4weight_return_cls.remark_lens
        rela_wiki_names_splited_list = get_content_info4weight_return_cls.rela_wiki_names_splited_list
        rela_wiki_names_str_list = get_content_info4weight_return_cls.rela_wiki_names_str_list
        is_bind_wiki_list = get_content_info4weight_return_cls.is_bind_wiki_list
        hash_obj_designer_flow = get_content_info4weight_return_cls.hash_obj_designer_flow
        hash_main_service_area = get_content_info4weight_return_cls.hash_main_service_area

        start_time = time.time()
        interact_time_np, before_last_year_flag, timeWeight, favoritesNumsNp, favoriteWeightNp = CalcWeight.get_interact_time_weight(
            params_collect_cls.ab_test, query, np.array(
                favoriteNums, dtype="float32"), np.array(commentNums, dtype="float32"),
            np.array(likeNums, dtype="float32"), np.array(publishTime, dtype="float32"))
        end_time = time.time()
        # log.logger.debug("before_last_year_flag"+str(before_last_year_flag))
        log.logger.info("计算 单位交互比 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 内容绑定wiki相关性
        start_time4jaccard_wiki = time.time()
        jaccard4content_wiki_name_np, full_match_wiki_name_np, split_match_wiki_name_np = CalcWeight.JaccardSim4WikiNames(
            query, query_splited_list,
            rela_wiki_names_splited_list,
            rela_wiki_names_str_list)
        end_time4jaccard_wiki = time.time()
        used_time4jaccard_wiki = end_time4jaccard_wiki - start_time4jaccard_wiki
        log.logger.info("计算 wiki 标题 的Jaccard Sim 运行时间：{:.10f} s".format(
            used_time4jaccard_wiki) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time4jaccard_content = time.time()
        jaccard4content_np, full_match_np, split_match_np = CalcWeight.JaccardSim4content(query, query_splited_list,
                                                                                          contents_splited_list,
                                                                                          contents_str_list)
        end_time4jaccard_content = time.time()
        used_time4jaccard_content = end_time4jaccard_content - start_time4jaccard_content
        log.logger.info("计算 文本内容 的Jaccard Sim 运行时间：{:.10f} s".format(
            used_time4jaccard_content) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 获取内容id 与 Jaccard Sim 的值
        hash_obj_id_jaccard_sim = {}
        hash_obj_id_jaccard_split_match_sim = {}

        for obj_id, jaccard_sim, split_match in zip(objIds, jaccard4content_np, split_match_np):
            if len(obj_id) > 0:
                hash_obj_id_jaccard_sim[obj_id] = jaccard_sim
                hash_obj_id_jaccard_split_match_sim[obj_id] = split_match

        start_time4wiki = time.time()
        wiki_score_np = CalcWeight.get_wiki_score_weight(wiki_score_list, before_last_year_flag, jaccard4content_np)
        end_time4wiki = time.time()
        used_time4wiki = end_time4wiki - start_time4wiki
        log.logger.info("计算 wiki分数 运行时间：{:.10f} s".format(
            used_time4wiki) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time4pool = time.time()
        pool_score_np = CalcWeight.get_pool_score_weight(pool_data_list, jaccard4content_np)
        end_time4pool = time.time()
        used_time4pool = end_time4pool - start_time4pool
        log.logger.info("计算 流量池分数 运行时间：{:.10f} s".format(
            used_time4pool) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        admin_score_np = CalcWeight.getAdminScoreWeight(
            np.array(adminScores, "float32"))
        end_time = time.time()
        log.logger.info("计算 后台质量分数 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        jaccard4admin_tag_np = CalcWeight.JaccardSim4admin_tag(query, query_splited_list, contentAdminTags)
        end_time = time.time()
        log.logger.info("计算 后台标签 的Jaccard Sim 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        title_lens_weight_np = CalcWeight.get_title_length_weight(np.array(title_lens, dtype=np.float))
        end_time = time.time()
        log.logger.info("计算 文本标题长度 的权重 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        start_time = time.time()
        remark_lens_weight_np = CalcWeight.get_remark_length_weight(np.array(remark_lens, dtype=np.float))
        # print("remark_lens_weight_np", remark_lens_weight_np)
        end_time = time.time()
        log.logger.info("计算 文本描述长度 的权重 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        matching_return_cls = matching_return()
        matching_return_cls.jaccard4content_np = jaccard4content_np
        matching_return_cls.jaccard4admin_tag_np = jaccard4admin_tag_np
        matching_return_cls.interact_time_np = interact_time_np
        matching_return_cls.admin_score_np = admin_score_np
        matching_return_cls.pool_score_np = pool_score_np
        matching_return_cls.wiki_score_np = wiki_score_np
        matching_return_cls.user_types_np = np.array(user_types)
        matching_return_cls.hash_obj_user_type = hash_obj_user_type
        matching_return_cls.hash_obj_id_jaccard_sim = hash_obj_id_jaccard_sim
        matching_return_cls.hash_obj_related_wiki = hash_obj_related_wiki
        matching_return_cls.full_match_np = full_match_np
        matching_return_cls.split_match_np = split_match_np
        matching_return_cls.title_lens_weight_np = title_lens_weight_np
        matching_return_cls.remark_lens_weight_np = remark_lens_weight_np
        matching_return_cls.time_weight_np = timeWeight
        matching_return_cls.favorites_nums_np = favoritesNumsNp
        matching_return_cls.pool_data_list = pool_data_list
        matching_return_cls.jaccard4content_wiki_name_np = jaccard4content_wiki_name_np
        matching_return_cls.hash_obj_id_jaccard_split_match_sim = hash_obj_id_jaccard_split_match_sim
        matching_return_cls.is_bind_wiki_np = np.array(is_bind_wiki_list)
        matching_return_cls.favorite_weight_np = favoriteWeightNp
        matching_return_cls.hash_obj_designer_flow = hash_obj_designer_flow
        matching_return_cls.hash_main_service_area = hash_main_service_area

        return matching_return_cls


    @classmethod
    def get_information_amounts(cls, params_collect_cls, esRecallFaissMapIdsFilter, hashEsDataInfos, query, init_request_global_var):

        start_time4len_sim_weight = time.time()

        objIds = FaissMap.getObjIdsByIds(esRecallFaissMapIdsFilter, True, True)

        # 获取内容相关数据 用于权重计算
        contents_lens_list = DataProcess.get_content_len(objIds, hashEsDataInfos)

        information_amounts_np = CalcWeight.calc_information_amounts(query, np.array(contents_lens_list))
        end_time4len_sim_weight = time.time()
        log.logger.info("计算 cosine_weight 运行时间：{:.10f} s".format(
            end_time4len_sim_weight - start_time4len_sim_weight) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
        return information_amounts_np
