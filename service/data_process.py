# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 14:59
@Desciption :  数据处理
'''

import numpy as np
import os
import sys
import time
os.chdir(sys.path[0])
sys.path.append("../")
from service.Common import Common
from designer.designer_pool import DesignerPool
from db.hhzMember.Member import Member
from service.Weight.calc_weight import CalcWeight
from service.Parmas.get_content_info4weight_return import get_content_info4weight_return
from common.common_log import Logger, get_log_name
import configparser
from service.Init.InitService import InitService

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

all_faiss_num = int(root_config["Recall"]["all_faiss_num"].strip())

designer_pool = DesignerPool()
# 设计师流量池的数据
designer_pool_dict, _, pool_set_sorted = designer_pool.get_designer_pool()
designer_uid_set = set(list(designer_pool_dict.keys()))

wiki_score_dict = {0: 0.9,  # 普通用户
                   2: 0.8,  # 认证设计师用户
                   1: 0.7,  # 品牌用户
                   3: 1,  # 个人V认证，例如，冯老板
                   4: 0.6,  # 未认证设计师
                   6: 0.7  # 装修公司
                   }

wiki_score_dict_keys = wiki_score_dict.keys()

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class DataProcess(object):
    SPLIT_WORDS_SIGN = "奰"

    @classmethod
    def get_content_infos4weight_calc4xgb(self, objIds, hashEsDataInfos):
        '''
        @param ：
            objIds: 内容id
            hashEsDataInfos： es中的数据
        @Description: 获取内容相关数据 用于权重计算
        '''
        # 内容id 属于用户的类型
        hash_obj_user_type = {}

        # 内容id 是否绑定了 wiki
        hash_obj_related_wiki = {}

        # 设计师 主服务地区
        hash_main_service_area = {}

        contents_splited_list = []

        contents_str_list = []

        likeNums = []

        favoriteNums = []

        commentNums = []

        publishTime = []

        adminScores = []

        wiki_score_list = []

        # 内容id 标题
        hash_obj_split_title_for_ai = {}

        # 内容id 描述
        hash_obj_split_remark_for_ai = {}

        for objId in objIds:
            if objId and objId in hashEsDataInfos:
                # 如果内容中包含wiki,则需要加权操作
                if "is_relate_wiki" in hashEsDataInfos[objId] and "user_type" in hashEsDataInfos[objId]:
                    is_relate_wiki = int(hashEsDataInfos[objId]["is_relate_wiki"])
                    user_type = int(hashEsDataInfos[objId]["user_type"])
                    # 对wiki内容，进行加权
                    # 内容id 是否 关联 wiki  用于业务排序 使用
                    is_relate_wiki_flag = 0

                    # if is_relate_wiki == 1 and query in adminTags:
                    if is_relate_wiki == 1:
                        if user_type in wiki_score_dict_keys:
                            is_relate_wiki_flag = 1
                            wiki_score = wiki_score_dict[user_type]
                        else:
                            wiki_score = 0
                    else:
                        wiki_score = 0

                    wiki_score_list.append(wiki_score)
                    # 内容id 是否 关联 wiki  用于业务排序 使用
                    hash_obj_related_wiki[objId] = is_relate_wiki_flag
                else:
                    # 没有数据的时候，直接设置为大盘水平，这个情况理论上不会出现，仅作为容错看待。
                    hash_obj_related_wiki[objId] = 0
                    wiki_score_list.append(0)

                # 设计师流量池的操作，看内容是是否是设计师的内容
                user_type = 0
                if "uid" in hashEsDataInfos[objId]:
                    if "user_type" in hashEsDataInfos[objId]:
                        user_type = int(hashEsDataInfos[objId]["user_type"])

                hash_obj_user_type[objId] = user_type

                if "like" in hashEsDataInfos[objId]:
                    likeNums.append(hashEsDataInfos[objId]["like"])
                else:
                    likeNums.append(0)
                #
                if "favorite" in hashEsDataInfos[objId]:
                    favoriteNums.append(hashEsDataInfos[objId]["favorite"])
                else:
                    favoriteNums.append(0)
                #
                if "comment" in hashEsDataInfos[objId]:
                    commentNums.append(hashEsDataInfos[objId]["comment"])
                else:
                    commentNums.append(0)
                #
                if "publish_time" in hashEsDataInfos[objId]:
                    publishTime.append(hashEsDataInfos[objId]["publish_time"])
                else:
                    publishTime.append(0)
                #
                if "admin_score" in hashEsDataInfos[objId]:
                    adminScores.append(self.transform_admin_score(objId, hashEsDataInfos[objId]["admin_score"]))
                else:
                    adminScores.append(0)

                # 格式化 es 中 切好的 标题
                if "split_words_title_array" in hashEsDataInfos[objId]:
                    hash_obj_split_title_for_ai[objId] = hashEsDataInfos[objId]["split_words_title_array"]
                else:
                    hash_obj_split_title_for_ai[objId] = []

                # 格式化 es 中 切好的 描述
                if "split_words_remark_array" in hashEsDataInfos[objId]:
                    hash_obj_split_remark_for_ai[objId] = hashEsDataInfos[objId]["split_words_remark_array"]
                else:
                    hash_obj_split_remark_for_ai[objId] = []

                # 设计师主服务地区
                if "main_service_area" in hashEsDataInfos[objId]:
                    hash_main_service_area[objId] = hashEsDataInfos[objId]["main_service_area"]

                if "index_name" in hashEsDataInfos[objId]:
                    if hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_PHOTO:
                        title = ""
                        remark = ""
                        if "title" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["title"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["title"])
                            title = hashEsDataInfos[objId]["title"]

                        if "desc" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["desc"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["desc"])
                            remark = hashEsDataInfos[objId]["desc"]

                        contents_str_list.append(title + " " + remark)

                    elif hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_TOTAL_ARTICLE:
                        contents_str_list.append(hashEsDataInfos[objId]["title"])

                if "split_words_array" in hashEsDataInfos[objId]:
                    splitWordsForAiList = set()

                    if len(splitWordsForAiList) > 0:
                        contents_splited_list.append(list(splitWordsForAiList))
                    else:
                        contents_splited_list.append([""])
                else:
                    contents_splited_list.append([""])
            else:
                contents_str_list.append("")
                contents_splited_list.append([""])

                likeNums.append(0)
                favoriteNums.append(0)
                commentNums.append(0)
                publishTime.append(0)
                adminScores.append(0)
                wiki_score_list.append(0)


        get_content_info4weight_return_cls = get_content_info4weight_return()

        get_content_info4weight_return_cls.hash_obj_user_type = hash_obj_user_type
        get_content_info4weight_return_cls.hash_obj_related_wiki = hash_obj_related_wiki
        get_content_info4weight_return_cls.hash_main_service_area = hash_main_service_area
        get_content_info4weight_return_cls.contents_str_list = contents_str_list
        get_content_info4weight_return_cls.contents_splited_list = contents_splited_list
        get_content_info4weight_return_cls.publish_time = publishTime
        get_content_info4weight_return_cls.hash_obj_split_title_for_ai = hash_obj_split_title_for_ai
        get_content_info4weight_return_cls.hash_obj_split_remark_for_ai = hash_obj_split_remark_for_ai
        get_content_info4weight_return_cls.favorite_nums = favoriteNums
        get_content_info4weight_return_cls.like_nums = likeNums
        get_content_info4weight_return_cls.comment_nums = commentNums
        get_content_info4weight_return_cls.admin_scores = adminScores
        get_content_info4weight_return_cls.wiki_score_list = wiki_score_list

        return get_content_info4weight_return_cls

    @classmethod
    def get_content_infos4weight_calc(self, objIds, query, hashEsDataInfos, params_collect_cls):
        '''
        @param ：
            objIds: 内容id
            hashEsDataInfos： es中的数据
        @Description: 获取内容相关数据 用于权重计算
        '''
        wiki_score_list = []  # wiki数据
        pool_data_list = []  # 流量池数据
        likeNums = []
        favoriteNums = []
        commentNums = []
        #
        publishTime = []
        adminScores = []

        contents_splited_list = []
        contents_str_list = []
        contentAdminTags = []

        contents_lens = []

        # 用户身份列表
        user_types = []

        # 内容绑定wiki列表
        is_bind_wiki_list = []

        # 内容id 属于用户的类型
        hash_obj_user_type = {}

        # 内容id 属于 a b c设计师流量
        hash_obj_designer_flow = {}

        # 内容id 是否绑定了 wiki
        hash_obj_related_wiki = {}

        # 内容id 标题
        hash_obj_split_title_for_ai = {}

        # 内容id 描述
        hash_obj_split_remark_for_ai = {}

        # 标题长度
        title_lens = []

        # 描述长度
        remark_lens = []

        # 内容绑定 wiki 的名字
        rela_wiki_names_splited_list = []
        rela_wiki_names_str_list = []

        # 7日内曝光数
        exposure_num_list = []

        # 7日内交互数
        interact_num_list = []

        # 设计师 主服务地区
        hash_main_service_area = {}

        # starttime = time.time()
        for objId in objIds:
            zeroNum = 0
            if objId and objId in hashEsDataInfos:
                adminTags = set()
                if "admin_tag" in hashEsDataInfos[objId]:
                    adminTag = hashEsDataInfos[objId]["admin_tag"].strip()
                    if len(adminTag) > 0:
                        adminTagsTmp = adminTag.split(" ")
                        for adminTagTmp in adminTagsTmp:
                            adminTagTmp = adminTagTmp.strip()
                            if len(adminTagTmp) > 0:
                                adminTags.add(adminTagTmp)

                    if len(adminTags) > 0:
                        contentAdminTags.append(list(adminTags))
                    else:
                        contentAdminTags.append([""])
                else:
                    contentAdminTags.append([""])

                # 如果内容中包含wiki,则需要加权操作
                if "is_relate_wiki" in hashEsDataInfos[objId] and "user_type" in hashEsDataInfos[objId]:
                    is_relate_wiki = int(hashEsDataInfos[objId]["is_relate_wiki"])
                    user_type = int(hashEsDataInfos[objId]["user_type"])
                    # 对wiki内容，进行加权
                    # 内容id 是否 关联 wiki  用于业务排序 使用
                    is_relate_wiki_flag = 0

                    # if is_relate_wiki == 1 and query in adminTags:
                    if is_relate_wiki == 1:
                        if user_type in wiki_score_dict_keys:
                            is_relate_wiki_flag = 1
                            wiki_score = wiki_score_dict[user_type]
                        else:
                            wiki_score = 0
                    else:
                        wiki_score = 0

                    wiki_score_list.append(wiki_score)
                    # 内容id 是否 关联 wiki  用于业务排序 使用
                    hash_obj_related_wiki[objId] = is_relate_wiki_flag

                    is_bind_wiki_list.append(is_relate_wiki_flag)
                else:
                    # 没有数据的时候，直接设置为大盘水平，这个情况理论上不会出现，仅作为容错看待。
                    wiki_score_list.append(0)
                    hash_obj_related_wiki[objId] = 0
                    is_bind_wiki_list.append(0)

                # 设计师流量池的操作，看内容是是否是设计师的内容
                user_type = 0
                if "uid" in hashEsDataInfos[objId]:
                    uid = hashEsDataInfos[objId]["uid"]

                    if "user_type" in hashEsDataInfos[objId]:
                        user_type = int(hashEsDataInfos[objId]["user_type"])

                    user_types.append(user_type)

                    if uid in designer_uid_set and user_type == Member.AUTH_USER_DESIGNER:  # user_type == 2 是设计师内容
                        if query in adminTags:
                            pool_data = designer_pool_dict[uid]
                        else:
                            pool_data = CalcWeight.general_pool_data
                    else:
                        # 普通用户或者非设计师的数据，直接设置为c池，也可以理解成大盘水平,d池的设计师的数据是低于大盘水平的(设计师流量池的内容经常变动！4个变成3个，然后又3个变成4个)
                        pool_data = CalcWeight.general_pool_data
                    pool_data_list.append(pool_data)

                    hash_obj_designer_flow[objId] = pool_data
                else:
                    # 没有数据的时候，直接设置为c池
                    # print("no user_type data", hashEsDataInfos[objId], "----------------", objId)
                    pool_data_list.append(CalcWeight.general_pool_data)
                    user_types.append(user_type)
                    hash_obj_designer_flow[objId] = CalcWeight.general_pool_data

                hash_obj_user_type[objId] = user_type

                if "like" in hashEsDataInfos[objId]:
                    likeNums.append(hashEsDataInfos[objId]["like"])
                else:
                    likeNums.append(0)
                #
                if "favorite" in hashEsDataInfos[objId]:
                    favoriteNums.append(hashEsDataInfos[objId]["favorite"])
                else:
                    favoriteNums.append(0)
                #
                if "comment" in hashEsDataInfos[objId]:
                    commentNums.append(hashEsDataInfos[objId]["comment"])
                else:
                    commentNums.append(0)
                #
                if "publish_time" in hashEsDataInfos[objId]:
                    publishTime.append(hashEsDataInfos[objId]["publish_time"])
                else:
                    publishTime.append(0)
                #
                if "admin_score" in hashEsDataInfos[objId]:
                    adminScores.append(self.transform_admin_score(objId, hashEsDataInfos[objId]["admin_score"]))
                else:
                    adminScores.append(0)

                if "rela_wiki_names" in hashEsDataInfos[objId]:
                    rela_wiki_names_str_list.append(hashEsDataInfos[objId]["rela_wiki_names"])
                else:
                    rela_wiki_names_str_list.append("")

                # 7日内 曝光数
                if "exposure_num" in hashEsDataInfos[objId]:
                    exposure_num_list.append(hashEsDataInfos[objId]["exposure_num"])
                else:
                    exposure_num_list.append(0)

                # 设计师主服务地区
                if "main_service_area" in hashEsDataInfos[objId]:
                    hash_main_service_area[objId] = hashEsDataInfos[objId]["main_service_area"]

                if "rela_wiki_names_split_array" in hashEsDataInfos[objId]:
                    relaWikiNamesSplit = hashEsDataInfos[objId]["rela_wiki_names_split_array"]
                    relaWikiNamesSplitList = set()

                    if len(relaWikiNamesSplit) > 0:
                        for relaWikiNamesSplitItem in relaWikiNamesSplit:
                            relaWikiNamesSplitItem = relaWikiNamesSplitItem.strip()
                            if len(relaWikiNamesSplitItem) > 0:
                                relaWikiNamesSplitList.add(relaWikiNamesSplitItem)

                    if len(relaWikiNamesSplitList) > 0:
                        rela_wiki_names_splited_list.append(list(relaWikiNamesSplitList))
                    else:
                        rela_wiki_names_splited_list.append([""])
                else:
                    rela_wiki_names_splited_list.append([""])


                # # 格式化 es 中 切好的 标题
                # if "split_words_for_ai_title" in hashEsDataInfos[objId]:
                #     splitWordsForAiTitleList = hashEsDataInfos[objId]["split_words_for_ai_title"].split(self.SPLIT_WORDS_SIGN)
                #     splitWordsForAiTitleList = [item for item in splitWordsForAiTitleList if item != '']
                #
                #     hash_obj_split_title_for_ai[objId] = splitWordsForAiTitleList
                #
                # # 格式化 es 中 切好的 描述
                # if "split_words_for_ai_remark" in hashEsDataInfos[objId]:
                #     splitWordsForAiRemarkList = hashEsDataInfos[objId]["split_words_for_ai_remark"].split(self.SPLIT_WORDS_SIGN)
                #     splitWordsForAiRemarkList = [item for item in splitWordsForAiRemarkList if item != '']

                    # hash_obj_split_remark_for_ai[objId] = splitWordsForAiRemarkList

                if "index_name" in hashEsDataInfos[objId]:
                    if hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_PHOTO:
                        title = ""
                        remark = ""
                        if "title" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["title"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["title"])
                            title = hashEsDataInfos[objId]["title"]

                        if "desc" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["desc"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["desc"])
                            remark = hashEsDataInfos[objId]["desc"]
                        # else:
                        #     # contents.append("")

                        # min_lenth = min(len(title), len(remark))
                        # if min_lenth == 0:
                        #     min_lenth = max(len(title), len(remark))

                        # min_lenth = max(len(title), len(remark))
                        content_lenth = len(title) + len(remark)
                        contents_str_list.append(title + " " + remark)
                        contents_lens.append(content_lenth)

                        title_lens.append(len(title))
                        remark_lens.append(len(remark))

                    elif hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_TOTAL_ARTICLE:
                        # contents.append(hashEsDataInfos[objId]["title"])
                        contents_lens.append(len(hashEsDataInfos[objId]["title"]))
                        contents_str_list.append(hashEsDataInfos[objId]["title"])

                        title_lens.append(len(hashEsDataInfos[objId]["title"]))
                        remark_lens.append(0)

                        # 处理 fulltext 为
                        if "split_words_remark_array" in hashEsDataInfos[objId]:
                            descList = hashEsDataInfos[objId]["split_words_remark_array"]
                            descStr = "".join(descList)
                            hashEsDataInfos[objId]["desc"] = descStr

                if "split_words_array" in hashEsDataInfos[objId]:
                    splitWordsForAi = hashEsDataInfos[objId]["split_words_array"]
                    splitWordsForAiList = set()

                    if len(splitWordsForAi) > 0:
                        for splitItem in splitWordsForAi:
                            splitItem = splitItem.strip()
                            if len(splitItem) > 0:
                                splitWordsForAiList.add(splitItem)

                    if len(splitWordsForAiList) > 0:
                        contents_splited_list.append(list(splitWordsForAiList))
                    else:
                        contents_splited_list.append([""])
                else:
                    contents_splited_list.append([""])

            else:
                wiki_score_list.append(0)
                pool_data_list.append(pool_set_sorted[-1])
                likeNums.append(zeroNum)
                favoriteNums.append(zeroNum)
                commentNums.append(zeroNum)
                #
                publishTime.append(0)
                adminScores.append(0)
                contents_splited_list.append([""])
                contentAdminTags.append([""])
                contents_lens.append(1)
                contents_str_list.append("")

        get_content_info4weight_return_cls = get_content_info4weight_return()

        get_content_info4weight_return_cls.wiki_score_list = wiki_score_list
        get_content_info4weight_return_cls.pool_data_list = pool_data_list
        get_content_info4weight_return_cls.like_nums = likeNums
        get_content_info4weight_return_cls.favorite_nums = favoriteNums
        get_content_info4weight_return_cls.comment_nums = commentNums
        get_content_info4weight_return_cls.publish_time = publishTime
        get_content_info4weight_return_cls.admin_scores = adminScores
        get_content_info4weight_return_cls.contents_splited_list = contents_splited_list
        get_content_info4weight_return_cls.content_admin_tags = contentAdminTags
        get_content_info4weight_return_cls.contents_lens = contents_lens
        get_content_info4weight_return_cls.contents_str_list = contents_str_list
        get_content_info4weight_return_cls.user_types = user_types
        get_content_info4weight_return_cls.hash_obj_user_type = hash_obj_user_type
        get_content_info4weight_return_cls.hash_obj_related_wiki = hash_obj_related_wiki
        get_content_info4weight_return_cls.title_lens = title_lens
        get_content_info4weight_return_cls.remark_lens = remark_lens
        get_content_info4weight_return_cls.rela_wiki_names_splited_list = rela_wiki_names_splited_list
        get_content_info4weight_return_cls.rela_wiki_names_str_list = rela_wiki_names_str_list
        get_content_info4weight_return_cls.is_bind_wiki_list = is_bind_wiki_list
        get_content_info4weight_return_cls.hash_obj_designer_flow = hash_obj_designer_flow
        get_content_info4weight_return_cls.hash_main_service_area = hash_main_service_area
        # get_content_info4weight_return_cls.hash_obj_split_title_for_ai = hash_obj_split_title_for_ai
        # get_content_info4weight_return_cls.hash_obj_split_remark_for_ai = hash_obj_split_remark_for_ai

        return  get_content_info4weight_return_cls

    @classmethod
    def get_content_len(cls, objIds, hashEsDataInfos):
        '''
        @param ：
            objIds: 内容id
            hashEsDataInfos： es中的数据
        @Description: 获取内存长度数据
        '''

        contents_lens = []
        # starttime = time.time()
        for objId in objIds:
            if objId and objId in hashEsDataInfos:

                if "index_name" in hashEsDataInfos[objId]:
                    if hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_PHOTO:
                        title = ""
                        remark = ""
                        if "title" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["title"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["title"])
                            title = hashEsDataInfos[objId]["title"]

                        if "desc" in hashEsDataInfos[objId] and len(hashEsDataInfos[objId]["desc"]) > 0:
                            # contents.append(hashEsDataInfos[objId]["desc"])
                            remark = hashEsDataInfos[objId]["desc"]

                        # min_lenth = max(len(title), len(remark))
                        content_lenth = len(title) + len(remark)
                        contents_lens.append(content_lenth)

                    elif hashEsDataInfos[objId]["index_name"] == Common.INDEX_SEARCH_OP_TOTAL_ARTICLE:
                        # contents.append(hashEsDataInfos[objId]["title"])
                        contents_lens.append(len(hashEsDataInfos[objId]["title"]))
            else:
                contents_lens.append(1)

        return contents_lens


    @classmethod
    def mergeEsRecallData(self, run_task_for_hot_return_cls, is_use_area_designer, user_area, faiss_obj_ids, ab_test):
        """
        合并es召回的数据 并返回 es召回的内容id
        Args:
            goodIntentClassNotes: 优质意图 note 数据
            allNoteInfos: 全量非差 note 数据
            goodIntentClassTotalArticles: 优质意图 整屋，文章，指南 数据
            allTotalArticles: 全量非差 整屋，文章，指南  数据

        Returns: set

        """
        # 存放es召回的内容id
        allNoteInfos = run_task_for_hot_return_cls.allNoteInfos
        allTotalArticles = run_task_for_hot_return_cls.allTotalArticles
        designerNotes = run_task_for_hot_return_cls.designerNotes
        designerTotalArticles = run_task_for_hot_return_cls.designerTotalArticles
        bindWikiNotes = run_task_for_hot_return_cls.bindWikiNotes
        bindWikiTotalArticles = run_task_for_hot_return_cls.bindWikiTotalArticles
        vectNoteInfosByIds = run_task_for_hot_return_cls.vectNoteInfosByIds
        vectTotalArticleInfosByIds = run_task_for_hot_return_cls.vectTotalArticleInfosByIds
        areaDesignerNotes = run_task_for_hot_return_cls.areaDesignerNotes
        areaDesignerTotalArticles = run_task_for_hot_return_cls.areaDesignerTotalArticles
        wiki2HighNotes = run_task_for_hot_return_cls.wiki2HighNotes

        esRecallObjIds = set()
        hashEsDataInfos = {}

        if len(allNoteInfos["rows"]) > 0:
            ids = set()
            for allNoteInfo in allNoteInfos["rows"]:
                ids.add(allNoteInfo["id"])
                allNoteInfo["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                allNoteInfo["desc"] = allNoteInfo["desc"][:128]

                hashEsDataInfos[allNoteInfo["id"]] = allNoteInfo

            esRecallObjIds = esRecallObjIds | set(ids)


        if len(allTotalArticles["rows"]) > 0:
            ids = set()
            for allTotalArticle in allTotalArticles["rows"]:
                ids.add(allTotalArticle["id"])
                allTotalArticle["index_name"] = Common.INDEX_SEARCH_OP_TOTAL_ARTICLE

                hashEsDataInfos[allTotalArticle["id"]] = allTotalArticle

            esRecallObjIds = esRecallObjIds | set(ids)

        if len(designerNotes["rows"]) > 0:
            ids = set()
            for designerNote in designerNotes["rows"]:
                ids.add(designerNote["id"])
                designerNote["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                designerNote["desc"] = designerNote["desc"][:128]
                hashEsDataInfos[designerNote["id"]] = designerNote

            esRecallObjIds = esRecallObjIds | set(ids)

        if len(designerTotalArticles["rows"]) > 0:
            ids = set()
            for designerTotalArticle in designerTotalArticles["rows"]:
                ids.add(designerTotalArticle["id"])
                designerTotalArticle["index_name"] = Common.INDEX_SEARCH_OP_TOTAL_ARTICLE

                hashEsDataInfos[designerTotalArticle["id"]] = designerTotalArticle

            esRecallObjIds = esRecallObjIds | set(ids)

        if len(bindWikiNotes["rows"]) > 0:
            ids = set()
            for bindWikiNote in bindWikiNotes["rows"]:
                ids.add(bindWikiNote["id"])
                bindWikiNote["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                bindWikiNote["desc"] = bindWikiNote["desc"][:128]
                hashEsDataInfos[bindWikiNote["id"]] = bindWikiNote

            esRecallObjIds = esRecallObjIds | set(ids)

        if len(bindWikiTotalArticles["rows"]) > 0:
            ids = set()
            for bindWikiTotalArticle in bindWikiTotalArticles["rows"]:
                ids.add(bindWikiTotalArticle["id"])
                bindWikiTotalArticle["index_name"] = Common.INDEX_SEARCH_OP_TOTAL_ARTICLE

                hashEsDataInfos[bindWikiTotalArticle["id"]] = bindWikiTotalArticle

            esRecallObjIds = esRecallObjIds | set(ids)

        if is_use_area_designer == 1:
            if len(areaDesignerNotes["rows"]) > 0:
                ids = set()

                city_designer_ratio = 0.1
                if user_area in InitService.hash_city_designer_ratio and \
                        InitService.hash_city_designer_ratio[user_area] > 0.1:
                    city_designer_ratio = city_designer_ratio

                all_designer_note_num = int(root_config["Recall"]["designer_note_num"].strip())
                pagesize = int(all_designer_note_num * city_designer_ratio)

                has_add_area_uid = set()
                has_add_area_id = set()
                while True:
                    is_break = True
                    for areaDesignerNote in areaDesignerNotes["rows"]:
                        if areaDesignerNote["uid"] in has_add_area_uid or areaDesignerNote["id"]  in has_add_area_id:
                            continue

                        is_break = False

                        has_add_area_uid.add(int(areaDesignerNote["uid"]))
                        has_add_area_id.add(areaDesignerNote["id"])

                        ids.add(areaDesignerNote["id"])
                        areaDesignerNote["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                        areaDesignerNote["desc"] = areaDesignerNote["desc"][:128]
                        hashEsDataInfos[areaDesignerNote["id"]] = areaDesignerNote

                        if len(ids) >= pagesize:
                            break

                    has_add_area_uid = set()
                    if is_break:
                        break

                esRecallObjIds = esRecallObjIds | set(ids)

            if len(areaDesignerTotalArticles["rows"]) > 0:
                ids = set()

                all_designer_total_article_num = int(root_config["Recall"]["designer_total_article_num"].strip())

                city_designer_ratio = 0.1
                if InitService.hash_city_designer_ratio[user_area] > 0.1:
                    city_designer_ratio = city_designer_ratio

                pagesize = int(all_designer_total_article_num * city_designer_ratio)

                has_add_area_uid = set()
                has_add_area_id = set()

                while True:
                    is_break = True

                    for areaDesignerTotalArticle in areaDesignerTotalArticles["rows"]:
                        if areaDesignerTotalArticle["uid"] in has_add_area_uid or areaDesignerTotalArticle["id"]  in has_add_area_id:
                            continue

                        is_break = False

                        has_add_area_uid.add(int(areaDesignerTotalArticle["uid"]))
                        has_add_area_id.add(areaDesignerTotalArticle["id"])

                        ids.add(areaDesignerTotalArticle["id"])
                        areaDesignerTotalArticle["index_name"] = Common.INDEX_SEARCH_OP_TOTAL_ARTICLE

                        hashEsDataInfos[areaDesignerTotalArticle["id"]] = areaDesignerTotalArticle

                        if len(ids) >= pagesize:
                            break

                    has_add_area_uid = set()
                    if is_break:
                        break

                esRecallObjIds = esRecallObjIds | set(ids)

        if len(wiki2HighNotes["rows"]) > 0:
            ids = set()

            for wiki2HighNote in wiki2HighNotes["rows"]:
                ids.add(wiki2HighNote["id"])
                wiki2HighNote["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                wiki2HighNote["desc"] = wiki2HighNote["desc"][:128]
                hashEsDataInfos[wiki2HighNote["id"]] = wiki2HighNote

            esRecallObjIds = esRecallObjIds | set(ids)

        # faiss 的 数据 只取 整体 召回的一部分
        needFaissDataSet = set(faiss_obj_ids)

        if len(vectNoteInfosByIds["rows"]) > 0:
            ids = set()
            for vectNoteInfosById in vectNoteInfosByIds["rows"]:
                if vectNoteInfosById["id"] not in needFaissDataSet:
                    continue

                ids.add(vectNoteInfosById["id"])
                vectNoteInfosById["index_name"] = Common.INDEX_SEARCH_OP_PHOTO
                vectNoteInfosById["desc"] = vectNoteInfosById["desc"][:128]
                hashEsDataInfos[vectNoteInfosById["id"]] = vectNoteInfosById

            esRecallObjIds = esRecallObjIds | set(ids)

        if len(vectTotalArticleInfosByIds["rows"]) > 0:
            ids = set()
            for vectTotalArticleInfosById in vectTotalArticleInfosByIds["rows"]:
                if vectTotalArticleInfosById["id"] not in needFaissDataSet:
                    continue

                ids.add(vectTotalArticleInfosById["id"])
                vectTotalArticleInfosById["index_name"] = Common.INDEX_SEARCH_OP_TOTAL_ARTICLE

                hashEsDataInfos[vectTotalArticleInfosById["id"]] = vectTotalArticleInfosById

            esRecallObjIds = esRecallObjIds | set(ids)

        return esRecallObjIds, hashEsDataInfos

    @classmethod
    def mergeEsRecallDataByTime(self, allNoteInfosByTime, allTotalArticlesByTime):
        idsReverseSorted = []

        mergeAllDatas = []

        if "rows" in allNoteInfosByTime and len(allNoteInfosByTime["rows"]) > 0:
            mergeAllDatas.extend(allNoteInfosByTime["rows"])

        if "rows" in allTotalArticlesByTime and len(allTotalArticlesByTime["rows"]) > 0:
            mergeAllDatas.extend(allTotalArticlesByTime["rows"])

        if len(mergeAllDatas) > 0:
            ids = []
            publishTimes = []

            for mergeData in mergeAllDatas:
                publishTime = 0
                id = ""

                if "id" in mergeData:
                    id = mergeData["id"]
                ids.append(id)

                if "publish_time" in mergeData:
                    publishTime = mergeData["publish_time"]
                publishTimes.append(publishTime)

            publishTimesNp = np.array(publishTimes)
            idsNp = np.array(ids)

            publishTimesNpSortedIndex = publishTimesNp.argsort()

            idsReverseSorted = idsNp[publishTimesNpSortedIndex[::-1]].tolist()

        return idsReverseSorted

    @classmethod
    def transform_admin_score(self, objId, admin_score):
        """
        转换 内容的 分数  普通内容的分数  note 普通为30 转换为60
        Args:

        Returns: int

        """
        if int(objId[8]) == Common.PHOTO_TYPE_NUMBER :
            admin_score_int = int(admin_score)
            if admin_score_int < Common.NORMAL_ADMIN_SCORE:
                admin_score = Common.NORMAL_ADMIN_SCORE

        return admin_score

    # @classmethod
    # def mergeFaissRecallData(self, allFaissMapIds):
    #     """
    #     合并faiss召回的数据 并返回 faiss召回的内容id 和 faiss map id
    #     Args:
    #         allFaissMapIds: 全量 faiss 召回
    #
    #     Returns:
    #
    #     """
    #     # faiss召回的内容 id
    #     faissRecallObjIds = set()
    #     # faiss map id
    #     faissRecallMapIds = set()
    #
    #     if len(allFaissMapIds) > 0:
    #         faissRecallMapIds = faissRecallMapIds | allFaissMapIds
    #
    #     if len(faissRecallMapIds) > 0:
    #         # 通过faissMapIds 获取其关联的内容id
    #         faissRecallObjIds = set(sFaissMap.getObjIdsByIds(faissRecallMapIds))
    #
    #     return faissRecallObjIds, faissRecallMapIds


