# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/4/28 14:53
@Desciption : 用于调度搜索服务  数据召回 和 数据融合
'''
import os
import sys
import socket
from typing import KeysView
import requests
import time
import numpy as np
import configparser
from sklearn.metrics.pairwise import cosine_similarity
os.chdir(sys.path[0])
sys.path.append("../")

from common.common_log import Logger, get_log_name
from service.faiss_map.FaissMap import FaissMap as sFaissMap
from cache.SearchContent import SearchContent as cSearchContent
from service.TaskThread import TaskThread
from service.Weight.match_operate import MatchingOp
from service.Weight.related_rank import RelatedRank
from service.Weight.comprehensive_rank import ComprehensiveRank
from service.interpose_record import InterposeRecord
from service.data_process import DataProcess
from service.Parmas.es_recall import EsRecallParams
from service.Common import Common
from service.Parmas.params_build import ParamsBuild
from service.Parmas.comprehensive_rank_ranking import comprehensive_rank_ranking
from service.Weight.business_rank import BusinessRanking
from service.Parmas.ranking_all import ranking_all
from service.Init.InitService import InitService
from service.Weight.calc_weight import CalcWeight
from cache.EsResult import EsResult
from coarse_graind_ranking.coarse_graind_rank import CoarseGraindRank
from service.Weight.coarse_graind_rank import FilterCoarseGraindRank

dir_name = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(dir_name, '../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

match_score_log_len = int(root_config["log"]['match_score_log_len'])

'''resource_config中的配置'''
search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)
use_gray_redis = False if search_opt_common_config["which_redis"]["use_gray_redis"] == "False" else True
use_redis_cache = False if search_opt_common_config["which_redis"]["use_redis_cache"] == "False" else True
level = search_opt_common_config["log"]["level"]

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)
hostname = socket.gethostname()

coarse_graind_rank = CoarseGraindRank()

class MainSearch(object):
    # 需要走意图查询的 意图
    needRunIntentClass = [
        "装修知识",
        "找服务",
        "找商品",
    ]

    def __init__(self):
        pass

    def SearchByTime(self, params_collect_cls, init_resource_cls, init_request_global_var):
        """
        时间排序 优先从缓存中获取数据，缓存数据不存在，重新生成。缓存有效期20分钟
        根据搜索词返回搜索结果， es召回数据 按照时间排序 获取最终结果并将结果生成缓存。
        Args:
            page:  页数
            query: 搜索词
            pagesize:条数
            searchCacheKey: redis 缓存 key
        Returns:

        """
        page = params_collect_cls.page
        pagesize = params_collect_cls.pagesize
        query = params_collect_cls.query_trans
        is_owner = params_collect_cls.is_owner
        content_types = params_collect_cls.content_types
        searchCacheKey = params_collect_cls.searchCacheKey
        # ab_test: ab test
        ab_test = params_collect_cls.ab_test

        start = (page - 1) * pagesize
        end = start + pagesize - 1

        # redis 缓存 key
        if searchCacheKey:
            searchKey = searchCacheKey
        else:
            # 缓存key
            searchKey = ParamsBuild.buildSearchKey(init_resource_cls, query, Common.SEARCH_TYPE_TIME_STR, is_owner, content_types, ab_test)

        if use_redis_cache:  # 如果use_redis_cache，则使用redis缓存
            searchContent = cSearchContent.getSearchContentInTime(init_resource_cls, searchKey, start, end)

            # 假设查询结果为空时， 那么为redis中写入empty 用于表示已经查过了，不再重复查询

            if len(searchContent) == 1 and searchContent[0] == "empty":
                return {
                    "rows": [],
                    "search_key": searchKey
                }
            elif len(searchContent) > 0:
                # 缓存有数据直接返回
                return {
                    "rows": searchContent,
                    "search_key": searchKey
                }

        # 获取搜索词的向量数据
        queryVec = None
        # queryVec = Common.getKeywordVect(
        #     init_resource_cls, query, init_request_global_var)

        # 获取切词结果 同义词结果 意图分类数据
        intentClass, splitWords, synonym_map, terms_weight, faiss_obj_ids_map, user_profile_info, user_click_seq_historys, \
        user_favorite_seq_historys = TaskThread.getQueryVecAndSplitWordAndFaissData(queryVec,
                                                                                                params_collect_cls,
                                                                                                init_resource_cls,
                                                                                                init_request_global_var)
        # 初始化es 召回参数类
        es_recall_params_cls = EsRecallParams()

        es_recall_params_cls.splitWords = splitWords
        es_recall_params_cls.synonym_map = synonym_map
        es_recall_params_cls.intentClass = intentClass
        es_recall_params_cls.terms_weight = terms_weight

        # 启动线程召回数据
        allNoteInfosByTime, allTotalArticlesByTime = TaskThread.runTaskForTime(params_collect_cls, init_resource_cls, init_request_global_var, es_recall_params_cls)

        # 排序好的内容ids
        sortedObjIds = DataProcess.mergeEsRecallDataByTime(allNoteInfosByTime, allTotalArticlesByTime)

        # 获取前一定条数
        sortedObjIds = sortedObjIds[:int(root_config["SearchServer"]["time_show_total_num"])]

        # 插入干预记录
        sortedObjIds = InterposeRecord.insertInterposeRecord(init_resource_cls, query, Common.SEARCH_TYPE_TIME_STR, sortedObjIds)

        # 存在排序好的内容id 生成缓存
        if len(sortedObjIds) > 0:
            formatData = {}

            for index, sortedObjId in enumerate(sortedObjIds):
                formatData[sortedObjId] = index
            # 插入缓存
            cSearchContent.addSearchContentInTime(init_resource_cls, searchKey, formatData)

            # end redis是前闭后闭  python截取是前闭后开
            return {
                "rows": sortedObjIds[start: end + 1],
                "search_key": searchKey
            }
        else:
            # 数据为空时 插入redis空的信息 防止重复查询
            formatData = {"empty": 0}
            cSearchContent.addSearchContentInTime(init_resource_cls, searchKey, formatData)
            return {
                "rows": [],
                "search_key": searchKey
            }


    def get_sorted_obj_ids(self, rank_scores, esRecallFaissMapIdsFilter, esRecallObjIds):
        """
        获取 排序好的 内容ids
        Args:
            rank_scores: 总体内容排序分数
            esRecallFaissMapIdsFilter: faiss map ids
        Returns:

        """
        # 获取按照相似度值排序 的索引位置
        sortIndex = np.argsort(-rank_scores)

        # 获取排序后的FaissMapId
        sortedFaissMapIds = esRecallFaissMapIdsFilter[sortIndex]

        # 基于FaissMapId 获取排序好的内容id
        sortedObjIds = sFaissMap.getObjIdsByIds(sortedFaissMapIds)

        # 和es召回的数据做比较 兼容 某些内容id 没有在faiss表里存在
        diffObjIds = set(esRecallObjIds) - set(sortedObjIds)
        sortedObjIds.extend(list(diffObjIds))

        # 获取按照相似度值排序的前n条
        sortedObjIds = sortedObjIds[: int(
            root_config["SearchServer"]["hot_show_total_num"])]

        return sortedObjIds

    def get_sorted_obj_ids_new(self, rank_scores, esRecallObjIdsList_TopK):
        """
        获取 排序好的 内容ids
        Args:
            rank_scores: 总体内容排序分数
            esRecallObjIdsList_TopK:  es 召回 的 内容 ids
        Returns:

        """
        # 获取按照相似度值排序 的索引位置
        sortIndex = np.argsort(-rank_scores)

        # 获取排序后的 内容 id 列表
        sortedObjIds = esRecallObjIdsList_TopK[sortIndex]

        # 获取按照相似度值排序的前n条
        sortedObjIds = sortedObjIds[: int(
            root_config["SearchServer"]["hot_show_total_num"])]

        return sortedObjIds

    def is_use_nick_recall(self, query):
        """
            是否使用昵称召回策略
        """
        if query not in InitService.all_brand_name:
            return 1
        else:
            return 0

    # 根据搜索词 进行 向量召回
    def VecRecall(self, query, num, init_resource_cls, unique_str):
        queryVecFineTure = Common.getKeywordNewVect(init_resource_cls, query, unique_str)

        similars = []
        obj_ids = []

        if queryVecFineTure is not None:
            similars, obj_ids = InitService.SearchANN.search(queryVecFineTure, num)

        return similars, obj_ids

    def SearchByHot(self, params_collect_cls, init_resource_cls, init_request_global_var):
        """
        contents
        rank阶段的代码
        热度排序 优先从缓存中获取数据，缓存数据不存在，重新生成。缓存有效期20分钟
        根据搜索词返回搜索结果， 基于 faiss召回数据 和 es召回数据 然后根据搜索词向量数据 和 内容向量数据做点积之后根据相似度排序
        获取最终结果并将结果生成缓存。
        Args:
        Returns:

        """
        # page:  页数
        page = params_collect_cls.page
        # pagesize:条数
        pagesize = params_collect_cls.pagesize
        # query: 搜索词
        query = params_collect_cls.query_trans
        # is_owner: 是否只看住友
        is_owner = params_collect_cls.is_owner
        # content_types: 内容类型
        content_types = params_collect_cls.content_types
        # searchCacheKey: redis 缓存 key
        searchCacheKey = params_collect_cls.searchCacheKey
        # ab_test: ab test
        ab_test = params_collect_cls.ab_test
        # 是否使用昵称召回策略
        params_collect_cls.is_use_nick_recall = self.is_use_nick_recall(query)
        # 用户所在地区
        user_area = params_collect_cls.user_area
        # 是否使用地区设计师策略
        is_use_area_designer = params_collect_cls.is_use_area_designer

        # redis 开始位置
        start = (page - 1) * pagesize
        # redis 结束位置
        end = start + pagesize - 1

        # redis 缓存 key
        if searchCacheKey:
            searchKey = searchCacheKey
        else:
            if ab_test == 21 or ab_test == 31:
                # 缓存key
                searchKey = ParamsBuild.buildSearchKey(init_resource_cls, query, Common.SEARCH_TYPE_HOT_STR, is_owner,
                                                       content_types, ab_test, params_collect_cls.uid)
            else:
                # 缓存key
                searchKey = ParamsBuild.buildSearchKey(init_resource_cls, query, Common.SEARCH_TYPE_HOT_STR, is_owner, content_types, ab_test)

        # 缓存中拿数据
        if use_redis_cache:
            starttime = time.time()
            searchContent = cSearchContent.getSearchContentInHot(init_resource_cls, searchKey, start, end)
            endtime = time.time()
            log.logger.info("缓存中拿数据 的 运行时间：{:.10f} s".format(
                    endtime - starttime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            # 假设查询结果为空时， 那么为redis中写入empty 用于表示已经查过了，不再重复查询
            if len(searchContent) == 1 and searchContent[0] == "empty":
                return {
                    "rows": [],
                    "search_key": searchKey
                }
            elif len(searchContent) > 0:
                # 缓存有数据直接返回
                return {
                    "rows": searchContent,
                    "search_key": searchKey
                }

        # 获取搜索词的向量数据
        queryVec = None
        # queryVec = Common.getKeywordVect(init_resource_cls, query, init_request_global_var)

        # 获取切词结果 同义词结果 意图分类数据 向量召回内容ids
        intentClass, splitWords, synonym_map, terms_weight, faiss_obj_ids_map, user_profile_info, \
        user_click_seq_historys, user_favorite_seq_historys = TaskThread.getQueryVecAndSplitWordAndFaissData(queryVec, params_collect_cls, init_resource_cls, init_request_global_var)

        # faiss 召回的数据
        faiss_obj_ids = faiss_obj_ids_map["obj_ids"]

        # 初始化es 召回参数类
        es_recall_params_cls = EsRecallParams()

        es_recall_params_cls.splitWords = splitWords
        es_recall_params_cls.synonym_map = synonym_map
        es_recall_params_cls.intentClass = intentClass
        es_recall_params_cls.terms_weight = terms_weight
        es_recall_params_cls.ab_test = ab_test
        es_recall_params_cls.is_use_nick_recall = params_collect_cls.is_use_nick_recall
        es_recall_params_cls.user_area = params_collect_cls.user_area
        es_recall_params_cls.is_use_area_designer = params_collect_cls.is_use_area_designer

        # 启动线程召回数据
        starttime4recall = time.time()
        run_task_for_hot_return_cls = TaskThread.runTaskForHot(
            params_collect_cls, init_resource_cls, init_request_global_var, es_recall_params_cls, faiss_obj_ids_map)

        endtime4recall = time.time()
        used_time4recall = endtime4recall - starttime4recall
        log.logger.info("es 召回 + faiss 召回 运行时间：{:.10f} s".format(
            used_time4recall) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
        if used_time4recall > 1:
            err_log.logger.error(hostname + " es 召回 + faiss 召回 运行时间：{:.10f} s".format(
                used_time4recall) + " uid:" + params_collect_cls.uid + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 合并es召回数据
        start_time4merge = time.time()
        esRecallObjIds, hashEsDataInfos = DataProcess.mergeEsRecallData(run_task_for_hot_return_cls,
                                                                        is_use_area_designer, user_area, faiss_obj_ids, ab_test)

        end_time4merge = time.time()
        use_time4merge = end_time4merge - start_time4merge
        log.logger.info("es合并es召回数据 运行时间：{:.10f} s".format(use_time4merge))

        # # 存储es查询的结果缓存
        # esResultKey = ParamsBuild.buildEsResultCacheKey(query, Common.SEARCH_TYPE_HOT_STR, is_owner,
        #                                                content_types)
        #
        # esResultHashData = EsResult.getEsResult(init_resource_cls, esResultKey)
        #
        # isNeedGetEs = True
        # esRecallObjIds = []
        # hashEsDataInfos = {}
        #
        # if "is_empty" in esResultHashData:
        # # if "is_empty" in esResultHashData or not use_redis_cache:
        #     isNeedGetEs = False
        # else:
        #     esRecallObjIds = set(esResultHashData[EsResult.ES_RECALL_OBJ_IDS]) if EsResult.ES_RECALL_OBJ_IDS in esResultHashData else set()
        #     hashEsDataInfos = esResultHashData[EsResult.HASH_ES_DATA_INFOS] if EsResult.HASH_ES_DATA_INFOS in esResultHashData else {}
        #
        # if isNeedGetEs:
        # # 初始化es 召回参数类
        #     es_recall_params_cls = EsRecallParams()
        #
        #     es_recall_params_cls.splitWords = splitWords
        #     es_recall_params_cls.synonym_map = synonym_map
        #     es_recall_params_cls.intentClass = intentClass
        #     es_recall_params_cls.terms_weight = terms_weight
        #     es_recall_params_cls.ab_test = ab_test
        #     es_recall_params_cls.is_use_nick_recall = params_collect_cls.is_use_nick_recall
        #     es_recall_params_cls.user_area = params_collect_cls.user_area
        #     es_recall_params_cls.is_use_area_designer = params_collect_cls.is_use_area_designer
        #
        #     # 启动线程召回数据
        #     starttime4recall = time.time()
        #     run_task_for_hot_return_cls = TaskThread.runTaskForHot(
        #             params_collect_cls, init_resource_cls, init_request_global_var, es_recall_params_cls, faiss_obj_ids_map)
        #
        #     endtime4recall = time.time()
        #     used_time4recall = endtime4recall - starttime4recall
        #     log.logger.info("es 召回 + faiss 召回 运行时间：{:.10f} s".format(used_time4recall) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
        #     if used_time4recall > 1:
        #         err_log.logger.error(hostname + " es 召回 + faiss 召回 运行时间：{:.10f} s".format(used_time4recall) + " uid:" + params_collect_cls.uid + " unique_str:" + init_request_global_var.UNIQUE_STR)
        #
        #     # 合并es召回数据
        #     start_time4merge = time.time()
        #     esRecallObjIds, hashEsDataInfos = DataProcess.mergeEsRecallData(run_task_for_hot_return_cls, is_use_area_designer, user_area)
        #
        #     end_time4merge = time.time()
        #     use_time4merge = end_time4merge - start_time4merge
        #     log.logger.info("es合并es召回数据 运行时间：{:.10f} s".format(use_time4merge))
        #
        #     if len(esRecallObjIds) > 0:
        #         EsResult.addEsResult(init_resource_cls, esResultKey, esRecallObjIds, hashEsDataInfos)
        #     else:
        #         EsResult.addEsResult(init_resource_cls, esResultKey, esRecallObjIds, {"is_empty":1})

        sortedObjIds = []
        if len(esRecallObjIds) > 0:
            # matching 阶段，计算综合排序所需的相关得分
            starttime = time.time()

            matching_return_cls = MatchingOp.matching_for_xgb(params_collect_cls, esRecallObjIds,
                                                      hashEsDataInfos, query, init_request_global_var,
                                                      splitWords)

            # matching 阶段，得到综合排序所需的相关得分
            hash_obj_user_type = matching_return_cls.hash_obj_user_type
            hash_obj_id_jaccard_sim = matching_return_cls.hash_obj_id_jaccard_sim
            hash_main_service_area = matching_return_cls.hash_main_service_area
            hash_obj_related_wiki = matching_return_cls.hash_obj_related_wiki

            endtime = time.time()

            used_time = endtime - starttime
            log.logger.info(hostname + " matching 阶段的整体运行时间：{:.10f} s".format(
                used_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)
            if used_time > 1:
                err_log.logger.error(hostname + " matching 阶段的整体运行时间：{:.10f} s".format(
                    used_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            # 相关性粗排
            esRecallObjIdsList = list(esRecallObjIds)
            esRecallObjIdsNp = np.array(esRecallObjIdsList)

            starttime = time.time()
            similars = coarse_graind_rank.predict(
                query, esRecallObjIdsList, splitWords, matching_return_cls)
            endtime = time.time()
            usedXgbTime = endtime - starttime

            log.logger.info("粗排 运行时间：{:.10f} s".format(usedXgbTime))

            sortIndex = np.argsort(-similars)
            sortedObjIds = esRecallObjIdsNp[sortIndex]

            rank_data_num = 200

            if len(sortedObjIds) > 0:
                if ab_test == 21:
                    # 用户个性化 精排
                    sortedObjIds = CalcWeight.topK_opt_user_profile(query, params_collect_cls, init_request_global_var,
                                                                    sortedObjIds, hashEsDataInfos, splitWords, rank_data_num,
                                                                    user_profile_info, user_click_seq_historys, user_favorite_seq_historys)

                else:
                    sortedObjIds = CalcWeight.topK_opt(query, params_collect_cls, init_request_global_var, sortedObjIds, hashEsDataInfos, splitWords, rank_data_num)

            # 地区强制排序
            if user_area in InitService.hash_city_designer_ratio:
                sortedObjIds = BusinessRanking.ranking_area_designer(sortedObjIds, hash_obj_user_type,
                                                                     hash_obj_id_jaccard_sim,
                                                                     hash_main_service_area,
                                                                     user_area)
            else:
                # 业务排序
                sortedObjIds = BusinessRanking.ranking(intentClass, sortedObjIds, hash_obj_user_type,
                                                       hash_obj_id_jaccard_sim, hash_obj_related_wiki)

            # 内容打散
            sortedObjIds = Common.break_up_data_by_uid(sortedObjIds)

            log.logger.debug("搜索词：" + query)
            log.logger.debug("分词效果：" + str(splitWords))

            # 插入干预记录
            sortedObjIds = InterposeRecord.insertInterposeRecord(init_resource_cls, query, Common.SEARCH_TYPE_HOT_STR, sortedObjIds)

        # 返回结果
        return self.get_respond(init_resource_cls, start, end, sortedObjIds, searchKey,  params_collect_cls, init_request_global_var)

    def get_respond(self, init_resource_cls, start, end, sortedObjIds, searchKey, params_collect_cls, init_request_global_var):
        """
        将排序好的内容存入 redis 并返回当前页的数据
        Args:
            start:  开始位置
            end: 结束位置
            sortedObjIds: 排序好的内容id
            searchKey: 搜索缓存 key
        Returns:
        """
        # 存在排序好的内容id 生成缓存
        if len(sortedObjIds) > 0:
            formatData = {}

            for index, sortedObjId in enumerate(sortedObjIds):
                formatData[sortedObjId] = index
            # 插入缓存
            if use_redis_cache:
                cSearchContent.addSearchContentInHot(init_resource_cls, searchKey, formatData)

            # end redis是前闭后闭  python截取是前闭后开
            return {
                "rows": sortedObjIds[start: end + 1],
                "search_key": searchKey
            }
        else:
            # 数据为空时 插入redis空的信息 防止重复查询
            starttime = time.time()
            formatData = {"empty": 0}
            # 插入缓存
            if use_redis_cache:
                cSearchContent.addSearchContentInHot(init_resource_cls, searchKey, formatData)

            endtime = time.time()
            used_time = endtime - starttime
            log.logger.info(hostname + " respond 运行时间：{:.10f} s".format(
                used_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

            return {
                "rows": [],
                "search_key": searchKey
            }


if __name__ == '__main__':
    mainsearch = MainSearch()
    # print(mainsearch.SearchByHot(1, "客厅"))
    # contents_lens_np = mainsearch.calc_len_sim_weight(
    #     np.array([1, 2, 3, 4, 5, 6, 7, 19, 20, 22, 25]))
    # print(contents_lens_np)
