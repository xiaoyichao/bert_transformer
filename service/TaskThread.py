# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 11:11
@Desciption :  调度线程任务
'''
from service.Es.EsRecall import EsRecall
from queue import Queue
from service.Common import Common
from common.common_log import Logger, get_log_name
from service.RecallThread.GetAllNoteInfos import GetAllNoteInfos
from service.RecallThread.GetAllTotalArticles import GetAllTotalArticles
from service.RecallThread.GetAllFaissInfos import GetAllFaissInfos
from service.RecallThread.GetVectNoteInfosByIds import GetVectNoteInfosByIds
from service.RecallThread.GetVectTotalArticleInfosByIds import GetVectTotalArticleInfosByIds

from service.RecallThread.GetAllNoteInfosByTime import GetAllNoteInfosByTime
from service.RecallThread.GetAllTotalArticlesByTime import GetAllTotalArticlesByTime
from service.RecallThread.GetDesignerNotes import GetDesignerNotes
from service.RecallThread.GetAreaDesignerNotes import GetAreaDesignerNotes
from service.RecallThread.GetAreaDesignerTotalArticle import GetAreaDesignerTotalArticle
from service.RecallThread.GetBindWikiNotes import GetBindWikiNotes
from service.RecallThread.GetWiki2HighNotes import GetWiki2HighNotes
from service.RecallThread.GetDesignerTotalArticles import GetDesignerTotalArticles
from service.RecallThread.GetBindWikiTotalArticles import GetBindWikiTotalArticles
from service.RecallThread.GetIntentInfo import GetIntentInfo
from service.RecallThread.GetSplitAndSynAndCoreWods import GetSplitAndSynAndCoreWods

from service.RecallThread.GetUserClickSeqHistorys import GetUserClickSeqHistorys
from service.RecallThread.GetUserFavoriteSeqHistorys import GetUserFavoriteSeqHistorys
from service.RecallThread.GetUserProfileInfo import GetUserProfileInfo

from service.Init.InitService import InitService
from service.Parmas.run_task_for_hot_return import run_task_for_hot_return


import os

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

class TaskThread(object):
    @classmethod
    def runTaskForHot(self, params_collect_cls, init_resource_cls, init_request_global_var, es_recall_params_cls, faiss_obj_ids_map):
        """
        热门排序使用 启动线程召回数据， 从es召回 优质意图note，优质意图文章 指南 整屋，全量非差文章 指南 整屋，全量非差note，
        从faiss中召回 优质意图faiss数据，全量非差faiss数据
        Args:
            keyword: 搜索词
            intentClass： 意图类型
            is_owner: 是否是住友发布
            content_types: 内容类型

        Returns:

        """

        # query: 搜索词
        keyword = params_collect_cls.query_trans
        # is_owner: 是否只看住友
        is_owner = params_collect_cls.is_owner
        # content_types: 内容类型
        content_types = params_collect_cls.content_types

        # es 召回类
        EsRecallCls = EsRecall()
        # 存放线程召回的数据
        recallDataQueue = Queue()
        # 存放召回的线程
        recallThreads = []

        # 获取搜索过滤的标签
        search_filter_tags = Common.get_search_filter_tags(init_resource_cls)

        # 区分核心词 还是 限定词
        core_words, limit_words, core_synonym_words, limit_synonym_words = Common.get_core_and_limit_words(es_recall_params_cls.terms_weight, es_recall_params_cls.synonym_map)

        # 内容类型  值为空 代表全部 不做筛选
        if content_types != "":
            content_types_set = set(content_types.split(","))
        else:
            content_types_set = set()

        is_use_note_index, is_use_total_article_index = Common.get_use_content_index(content_types_set)

        es_recall_params_cls.keyword = keyword
        es_recall_params_cls.is_owner = is_owner
        es_recall_params_cls.content_types_set = content_types_set
        es_recall_params_cls.search_filter_tags = search_filter_tags
        es_recall_params_cls.core_words = core_words
        es_recall_params_cls.limit_words = limit_words
        es_recall_params_cls.core_synonym_words = core_synonym_words
        es_recall_params_cls.limit_synonym_words = limit_synonym_words

        # 初始化获取 全量非差note信息 的线程
        if len(content_types_set) == 0 or is_use_note_index:
            AllNoteInfosThreadId = GetAllNoteInfos.TYPE
            AllNoteInfosThread = GetAllNoteInfos(params_collect_cls, AllNoteInfosThreadId, recallDataQueue,
                                                 EsRecallCls,  es_recall_params_cls, init_resource_cls, init_request_global_var)
            AllNoteInfosThread.setDaemon(True)
            AllNoteInfosThread.start()
            recallThreads.append(AllNoteInfosThread)

            # 设计师内容
            DesignerNoteInfosThreadId = GetDesignerNotes.TYPE
            DesignerNoteInfosThread = GetDesignerNotes(params_collect_cls, DesignerNoteInfosThreadId, recallDataQueue,
                                                 EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                 init_request_global_var)
            DesignerNoteInfosThread.setDaemon(True)
            DesignerNoteInfosThread.start()
            recallThreads.append(DesignerNoteInfosThread)

            # 主服务地区 设计师内容
            if es_recall_params_cls.is_use_area_designer == 1 and es_recall_params_cls.user_area in InitService.hash_city_designer_ratio:
                AreaDesignerNoteInfosThreadId = GetAreaDesignerNotes.TYPE
                AreaDesignerNoteInfosThread = GetAreaDesignerNotes(params_collect_cls, AreaDesignerNoteInfosThreadId, recallDataQueue,
                                                           EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                           init_request_global_var)
                AreaDesignerNoteInfosThread.setDaemon(True)
                AreaDesignerNoteInfosThread.start()
                recallThreads.append(AreaDesignerNoteInfosThread)

            # 绑定 wiki 内容
            BindWikiNoteInfosThreadId = GetBindWikiNotes.TYPE
            BindWikiNoteInfosThread = GetBindWikiNotes(params_collect_cls, BindWikiNoteInfosThreadId, recallDataQueue,
                                                       EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                       init_request_global_var)
            BindWikiNoteInfosThread.setDaemon(True)
            BindWikiNoteInfosThread.start()
            recallThreads.append(BindWikiNoteInfosThread)

            # wiki 双高池 数据
            GetWiki2HighNotesThreadId = GetWiki2HighNotes.TYPE
            GetWiki2HighNotesThread = GetWiki2HighNotes(params_collect_cls, GetWiki2HighNotesThreadId, recallDataQueue,
                                                       EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                       init_request_global_var)
            GetWiki2HighNotesThread.setDaemon(True)
            GetWiki2HighNotesThread.start()
            recallThreads.append(GetWiki2HighNotesThread)

        if len(content_types_set) == 0 or is_use_total_article_index:
            # 初始化获取  全量非差文章，整屋，指南信息 的线程
            AllTotalArticlesThreadId = GetAllTotalArticles.TYPE
            AllTotalArticlesThread = GetAllTotalArticles(params_collect_cls, AllTotalArticlesThreadId,
                                                         recallDataQueue,
                                                         EsRecallCls,  es_recall_params_cls, init_resource_cls, init_request_global_var)
            AllTotalArticlesThread.setDaemon(True)
            AllTotalArticlesThread.start()
            recallThreads.append(AllTotalArticlesThread)

            # 设计师内容
            DesignerTotalArticleInfosThreadId = GetDesignerTotalArticles.TYPE
            DesignerTotalArticleInfosThread = GetDesignerTotalArticles(params_collect_cls, DesignerTotalArticleInfosThreadId, recallDataQueue,
                                                       EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                       init_request_global_var)
            DesignerTotalArticleInfosThread.setDaemon(True)
            DesignerTotalArticleInfosThread.start()
            recallThreads.append(DesignerTotalArticleInfosThread)

            # 主服务地区 设计师内容
            if es_recall_params_cls.is_use_area_designer == 1:
                AreaDesignerTotalArticleInfosThreadId = GetAreaDesignerTotalArticle.TYPE
                AreaDesignerTotalArticleInfosThread = GetAreaDesignerTotalArticle(params_collect_cls, AreaDesignerTotalArticleInfosThreadId,
                                                                   recallDataQueue,
                                                                   EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                                   init_request_global_var)
                AreaDesignerTotalArticleInfosThread.setDaemon(True)
                AreaDesignerTotalArticleInfosThread.start()
                recallThreads.append(AreaDesignerTotalArticleInfosThread)

            # 绑定 wiki 内容
            BindWikiTotalArticleInfosThreadId = GetBindWikiTotalArticles.TYPE
            BindWikiTotalArticleInfosThread = GetBindWikiTotalArticles(params_collect_cls, BindWikiTotalArticleInfosThreadId, recallDataQueue,
                                                       EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                       init_request_global_var)
            BindWikiTotalArticleInfosThread.setDaemon(True)
            BindWikiTotalArticleInfosThread.start()
            recallThreads.append(BindWikiTotalArticleInfosThread)

        if len(content_types_set) == 0 or is_use_note_index:
            if len(faiss_obj_ids_map["note"]) > 0:
                # 根据向量召回的内容id 获取es的数据 note索引
                GetVectNoteInfosByIdsThreadId = GetVectNoteInfosByIds.TYPE
                GetVectNoteInfosByIdsThread = GetVectNoteInfosByIds(params_collect_cls,
                                                                           GetVectNoteInfosByIdsThreadId,
                                                                       recallDataQueue,
                                                                       EsRecallCls, es_recall_params_cls, init_resource_cls,
                                                                       init_request_global_var, faiss_obj_ids_map["note"])
                GetVectNoteInfosByIdsThread.setDaemon(True)
                GetVectNoteInfosByIdsThread.start()
                recallThreads.append(GetVectNoteInfosByIdsThread)

        if len(content_types_set) == 0 or is_use_total_article_index:
            if len(faiss_obj_ids_map["not_note"]) > 0:
                # 根据向量召回的内容id 获取es的数据 非note索引
                GetVectTotalArticleInfosByIdsThreadId = GetVectTotalArticleInfosByIds.TYPE
                GetVectTotalArticleInfosByIdsThread = GetVectTotalArticleInfosByIds(params_collect_cls,
                                                                           GetVectTotalArticleInfosByIdsThreadId,
                                                                               recallDataQueue,
                                                                               EsRecallCls, es_recall_params_cls,
                                                                               init_resource_cls,
                                                                               init_request_global_var, faiss_obj_ids_map["not_note"])
                GetVectTotalArticleInfosByIdsThread.setDaemon(True)
                GetVectTotalArticleInfosByIdsThread.start()
                recallThreads.append(GetVectTotalArticleInfosByIdsThread)

        # 等待所有线程结束
        for recallThread in recallThreads:
            recallThread.join()

        # 返回数据初始化
        run_task_for_hot_return_cls = run_task_for_hot_return()

        # 分配线程召回数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)

            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetAllNoteInfos.TYPE:
                    run_task_for_hot_return_cls.allNoteInfos = recallInfo["data"]
                elif recallInfo["type"] == GetAllTotalArticles.TYPE:
                    run_task_for_hot_return_cls.allTotalArticles = recallInfo["data"]
                elif recallInfo["type"] == GetDesignerNotes.TYPE:
                    run_task_for_hot_return_cls.designerNotes = recallInfo["data"]
                elif recallInfo["type"] == GetDesignerTotalArticles.TYPE:
                    run_task_for_hot_return_cls.designerTotalArticles = recallInfo["data"]
                elif recallInfo["type"] == GetBindWikiNotes.TYPE:
                    run_task_for_hot_return_cls.bindWikiNotes = recallInfo["data"]
                elif recallInfo["type"] == GetBindWikiTotalArticles.TYPE:
                    run_task_for_hot_return_cls.bindWikiTotalArticles = recallInfo["data"]
                elif recallInfo["type"] == GetVectNoteInfosByIds.TYPE:
                    run_task_for_hot_return_cls.vectNoteInfosByIds = recallInfo["data"]
                elif recallInfo["type"] == GetVectTotalArticleInfosByIds.TYPE:
                    run_task_for_hot_return_cls.vectTotalArticleInfosByIds = recallInfo["data"]
                elif recallInfo["type"] == GetAreaDesignerNotes.TYPE:
                    run_task_for_hot_return_cls.areaDesignerNotes = recallInfo["data"]
                elif recallInfo["type"] == GetAreaDesignerTotalArticle.TYPE:
                    run_task_for_hot_return_cls.areaDesignerTotalArticles = recallInfo["data"]
                elif recallInfo["type"] == GetWiki2HighNotes.TYPE:
                    run_task_for_hot_return_cls.wiki2HighNotes = recallInfo["data"]

        return run_task_for_hot_return_cls

    @classmethod
    def runTaskForTime(self, params_collect_cls, init_resource_cls, init_request_global_var, es_recall_params_cls):
        """
        时间排序使用 启动线程召回数据， 从es召回  note ， 文章 整屋 指南
        Args:
            keyword: 搜索词
            is_owner: 是否是住友发布
            content_types: 内容类型

        Returns:

        """
        # 搜索词
        keyword = params_collect_cls.query
        # 是否只看住友 即 只看普通用户
        is_owner = params_collect_cls.is_owner
        # 内容类型
        content_types = params_collect_cls.content_types

        # es 召回类
        EsRecallCls = EsRecall()

        # 存放线程召回的数据
        recallDataQueue = Queue()

        # 存放召回的线程
        recallThreads = []

        # 获取搜索过滤的标签
        search_filter_tags = Common.get_search_filter_tags(init_resource_cls)

        # 区分核心词 还是 限定词
        core_words, limit_words, core_synonym_words, limit_synonym_words = Common.get_core_and_limit_words(
            es_recall_params_cls.terms_weight, es_recall_params_cls.synonym_map)

        if content_types != "":
            content_types_set = set(content_types.split(","))
        else:
            content_types_set = set()

        es_recall_params_cls.keyword = keyword
        es_recall_params_cls.is_owner = is_owner
        es_recall_params_cls.content_types_set = content_types_set
        es_recall_params_cls.search_filter_tags = search_filter_tags
        es_recall_params_cls.core_words = core_words
        es_recall_params_cls.limit_words = limit_words
        es_recall_params_cls.core_synonym_words = core_synonym_words
        es_recall_params_cls.limit_synonym_words = limit_synonym_words

        if len(content_types_set) == 0 or Common.PHOTO_TYPE in content_types_set or Common.ANSWER_TYPE in content_types_set or Common.VIDEO_TYPE in content_types_set:
            # 初始化获取 全量非差note信息 带时间字段 用于排序 的线程
            AllNoteInfosByTimeThreadId = GetAllNoteInfosByTime.TYPE
            AllNoteInfosByTimeThread = GetAllNoteInfosByTime(params_collect_cls, AllNoteInfosByTimeThreadId,
                                                             recallDataQueue,
                                                             EsRecallCls, es_recall_params_cls, init_resource_cls, init_request_global_var)
            AllNoteInfosByTimeThread.start()
            recallThreads.append(AllNoteInfosByTimeThread)

        if len(
                content_types_set) == 0 or Common.ARTICLE_TYPE in content_types_set or Common.BLANK_TYPE in content_types_set:
            # 初始化获取  全量非差文章，整屋，指南信息 带时间字段 用于排序 的线程
            AllTotalArticlesByTimeThreadId = GetAllTotalArticlesByTime.TYPE
            AllTotalArticlesByTimeThread = GetAllTotalArticlesByTime(params_collect_cls, AllTotalArticlesByTimeThreadId,
                                                                     recallDataQueue,
                                                                     EsRecallCls, es_recall_params_cls, init_resource_cls, init_request_global_var)
            AllTotalArticlesByTimeThread.start()
            recallThreads.append(AllTotalArticlesByTimeThread)

        # 等待所有线程结束
        for recallThread in recallThreads:
            recallThread.join()

        initEsReturnData = {
            "total": 0,
            "rows": [],
            "max_score": 0,
        }

        allNoteInfosByTime = initEsReturnData
        allTotalArticlesByTime = initEsReturnData

        # 分配线程召回数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)
            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetAllNoteInfosByTime.TYPE:
                    allNoteInfosByTime = recallInfo["data"]
                elif recallInfo["type"] == GetAllTotalArticlesByTime.TYPE:
                    allTotalArticlesByTime = recallInfo["data"]

        return allNoteInfosByTime, allTotalArticlesByTime

    @classmethod
    def getQueryVecAndSplitWordAndFaissData(cls, queryVec, params_collect_cls, init_resource_cls, init_request_global_var):
        """
        获取搜索词向量 切词结果 同义词结果 词权重 向量召回数据 启动线程获取结果
        Args:
            queryVecFineTure: 搜索词向量 dssm
            params_collect_cls: 参数收集类
            init_resource_cls: 资源类
            init_request_global_var: 全局变量类
        """

        # 存放线程召回的数据
        recallDataQueue = Queue()

        # 存放召回的线程
        recallThreads = []

        # 初始化获取 意图分类的 数据 的线程
        # GetIntentInfoThreadId = GetIntentInfo.TYPE
        # GetIntentInfoThread = GetIntentInfo(queryVec, params_collect_cls, GetIntentInfoThreadId,
        #                                     recallDataQueue,
        #                                     init_resource_cls,
        #                                     init_request_global_var)
        # GetIntentInfoThread.start()
        # recallThreads.append(GetIntentInfoThread)

        # 初始化获取 切词 同义词 词权重 用于排序 的线程
        GetSplitAndSynAndCoreWodsThreadId = GetSplitAndSynAndCoreWods.TYPE
        GetSplitAndSynAndCoreWodsThread = GetSplitAndSynAndCoreWods(params_collect_cls, GetSplitAndSynAndCoreWodsThreadId,
                                                                    recallDataQueue,
                                                                    init_resource_cls,
                                                                    init_request_global_var)
        GetSplitAndSynAndCoreWodsThread.start()
        recallThreads.append(GetSplitAndSynAndCoreWodsThread)


        # 初始化获取 向量召回数据 的线程
        if len(params_collect_cls.query_trans) > 0:
            GetAllFaissInfosThreadId = GetAllFaissInfos.TYPE
            GetAllFaissInfosThread = GetAllFaissInfos(params_collect_cls.query_trans,
                                                                        GetAllFaissInfosThreadId,
                                                                        recallDataQueue,
                                                                         params_collect_cls.ab_test,
                                                      init_request_global_var.UNIQUE_STR,
                                                      params_collect_cls.uid
                                                      )
            GetAllFaissInfosThread.start()
            recallThreads.append(GetAllFaissInfosThread)

        if params_collect_cls.ab_test == 21 or params_collect_cls.ab_test == 31:
            # 获取 当前用户 用户分群信息
            GetUserProfileInfoThreadId = GetUserProfileInfo.TYPE
            GetUserProfileInfoThread = GetUserProfileInfo(params_collect_cls, GetUserProfileInfoThreadId,
                                                                    recallDataQueue,
                                                                    init_resource_cls,
                                                                    init_request_global_var)
            GetUserProfileInfoThread.start()
            recallThreads.append(GetUserProfileInfoThread)

            # 获取 用户 有效点击 序列
            GetUserClickSeqHistorysThreadId = GetUserClickSeqHistorys.TYPE
            GetUserClickSeqHistorysThread = GetUserClickSeqHistorys(params_collect_cls, GetUserClickSeqHistorysThreadId,
                                                          recallDataQueue,
                                                          init_resource_cls,
                                                          init_request_global_var)
            GetUserClickSeqHistorysThread.start()
            recallThreads.append(GetUserClickSeqHistorysThread)

            # 获取 用户 收藏 序列
            GetUserFavoriteSeqHistorysThreadId = GetUserFavoriteSeqHistorys.TYPE
            GetUserFavoriteSeqHistorysThread = GetUserFavoriteSeqHistorys(params_collect_cls, GetUserFavoriteSeqHistorysThreadId,
                                                               recallDataQueue,
                                                               init_resource_cls,
                                                               init_request_global_var)
            GetUserFavoriteSeqHistorysThread.start()
            recallThreads.append(GetUserFavoriteSeqHistorysThread)

        # 等待所有线程结束
        for recallThread in recallThreads:
            recallThread.join()

        query_intent_list = []
        faiss_obj_ids_map = {
            "note" : [],
            "not_note" : [],
            "obj_ids" : []
        }
        split_words = []
        synonym_map = {}
        terms_weight = {}
        
        user_profile_info = {}
        user_click_seq_historys = []
        user_favorite_seq_historys = []

        # 分配线程获取数据
        for _ in range(len(recallThreads)):
            recallInfo = recallDataQueue.get(block=False)

            if recallInfo is not None and type(recallInfo) == dict and 'type' in recallInfo:
                if recallInfo["type"] == GetIntentInfo.TYPE:
                    query_intent_list = recallInfo["data"]

                elif recallInfo["type"] == GetSplitAndSynAndCoreWods.TYPE:
                    split_and_syn_and_core_wods = recallInfo["data"]

                    split_words = split_and_syn_and_core_wods["split_words"]
                    synonym_map = split_and_syn_and_core_wods["synonym_map"]
                    terms_weight = split_and_syn_and_core_wods["terms_weight"]

                elif recallInfo["type"] == GetAllFaissInfos.TYPE:
                    faiss_obj_ids_map = recallInfo["data"]

                elif recallInfo["type"] == GetUserProfileInfo.TYPE:
                    user_profile_info = recallInfo["data"]

                elif recallInfo["type"] == GetUserClickSeqHistorys.TYPE:
                    user_click_seq_historys = recallInfo["data"]

                elif recallInfo["type"] == GetUserFavoriteSeqHistorys.TYPE:
                    user_favorite_seq_historys = recallInfo["data"]

        return query_intent_list, split_words, synonym_map, terms_weight, faiss_obj_ids_map, \
               user_profile_info, user_click_seq_historys, user_favorite_seq_historys