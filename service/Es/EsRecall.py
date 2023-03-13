# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/4/28 15:15
@Desciption :  es数据召回 相关逻辑
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")
from es.Note import Note as esNote
from es.Note_v1 import Note_v1 as esNoteV1
from es.TotalArticle import TotalArticle as esTotalArticle
from es.TotalArticle_v1 import TotalArticle_v1 as esTotalArticleV1
from service.Common import Common
from service.Tag import Tag as sTag
import configparser
from service.Init.InitService import InitService

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)


class EsRecall(object):
    # SEARCH OP PHOTO 热门排序 字段
    # SEARCH_OP_PHOTO_HOT_FL = "id,title,desc,split_words_for_ai,admin_tag,like,favorite,comment,publish_time,admin_score"
    SEARCH_OP_PHOTO_HOT_FL = "id,title,desc,admin_tag,like,favorite,comment,publish_time,admin_score,user_type,uid,is_relate_wiki,rela_wiki_names,exposure_num,like_in_exposure,favorite_in_exposure,comment_in_exposure,share_in_exposure,main_service_area,split_words_array,split_words_title_array,split_words_remark_array,rela_wiki_names_split_array"

    # SEARCH OP TOTAL ARTICLE 热门排序 字段
    # SEARCH_OP_TOTAL_ARTICLE_HOT_FL = "id,title,desc,split_words_for_ai,admin_tag,like,favorite,comment,publish_time,admin_score"
    SEARCH_OP_TOTAL_ARTICLE_HOT_FL = "id,title,admin_tag,like,favorite,comment,publish_time,admin_score,user_type,uid,is_relate_wiki,rela_wiki_names,exposure_num,like_in_exposure,favorite_in_exposure,comment_in_exposure,share_in_exposure,main_service_area,split_words_array,split_words_title_array,split_words_remark_array,rela_wiki_names_split_array"

    def getDesignerNotes(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 设计师 note
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        all_designer_note_num = int(root_config["Recall"]["designer_note_num"].strip())

        city_designer_ratio = 0.1
        if es_recall_params_cls.user_area in InitService.hash_city_designer_ratio and \
                InitService.hash_city_designer_ratio[es_recall_params_cls.user_area] > 0.1:
            city_designer_ratio = city_designer_ratio

        pagesize = all_designer_note_num * (1 - city_designer_ratio)

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getDesignerNote(1, int(pagesize), fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getAreaDesignerNotes(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 地区 设计师 note
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        pagesize = int(root_config["Recall"]["area_designer_note_num"].strip())

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getAreaDesignerNote(1, pagesize, fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getBindWikiNotes(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 绑定了 wiki内容 note
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        pagesize = int(root_config["Recall"]["wiki_note_num"].strip())

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getBindWikiNote(1, pagesize, fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getWiki2HighNotes(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回  wiki 双高池 数据
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        pagesize = int(root_config["Recall"]["wiki_2_high_note_num"].strip())

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getWiki2HighNote(1, pagesize, fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getAllNoteInfos(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回非差note
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        pagesize = int(root_config["Recall"]["all_note_num"].strip())

        # 意图查询召回存在时，取少一点 意图查询召回不存在时，取多一点
        # if len(set(Common.needRunIntentClass) & set(es_recall_params_cls.intentClass)) == 0:
        #     pagesize = int(root_config["Recall"]["all_note_num_not_have_intent"].strip())

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getAllNoteForSearch(1, pagesize, fl,  es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getAllNoteInfosByTime(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        全量非差note信息 带时间字段 用于排序
        @Author: xuzhongjie
        Args:
            keyword: 原始搜索词
            splitWords: 切词结果
            SynonymWords: 同义词
            is_owner: 是否只看住友
            content_types_set: 内容类型
            search_filter_tags： 搜索过滤标签

        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        fl = "id,publish_time"

        return esNoteCls.getAllNoteByTimeForSearch(1, int(root_config["Recall"]["all_note_num_by_time"].strip()),  fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getAllTotalArticleInfosByTime(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        全量 total article 信息 带时间字段 用于排序
        @Author: xuzhongjie
        Args:
            keyword: 原始搜索词
            splitWords: 切词结果
            SynonymWords: 同义词
            is_owner: 是否只看住友
            content_types_set: 内容类型
            search_filter_tags： 搜索过滤标签

        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        fl = "id,publish_time"

        return esTotalArticleCls.getTotalArticleByTimeForSearch(1, int(root_config["Recall"]["all_total_article_num_by_time"].strip()), fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getDesignerTotalArticles(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 设计师 整屋 文章 指南
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        fl = self.SEARCH_OP_TOTAL_ARTICLE_HOT_FL

        all_designer_total_article_num = int(root_config["Recall"]["designer_total_article_num"].strip())

        city_designer_ratio = 0.1
        if es_recall_params_cls.user_area in InitService.hash_city_designer_ratio and \
                InitService.hash_city_designer_ratio[es_recall_params_cls.user_area] > 0.1:
            city_designer_ratio = city_designer_ratio

        pagesize = all_designer_total_article_num * (1 - city_designer_ratio)

        return esTotalArticleCls.getDesignerTotalArticle(1, int(pagesize), fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getAreaDesignerTotalArticles(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 地区 设计师 total article
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        fl = self.SEARCH_OP_TOTAL_ARTICLE_HOT_FL

        pagesize = int(root_config["Recall"]["area_designer_total_article_num"].strip())

        return esTotalArticleCls.getAreaDesignerTotalArticle(1, pagesize, fl, es_recall_params_cls, init_resource_cls,
                                                         init_request_global_var)

    def getBindWikiTotalArticles(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回 绑定wiki 整屋 文章 指南
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        fl = self.SEARCH_OP_TOTAL_ARTICLE_HOT_FL

        pagesize = int(root_config["Recall"]["wiki_total_article_num"].strip())

        return esTotalArticleCls.getBindWikiTotalArticle(1, pagesize, fl, es_recall_params_cls, init_resource_cls, init_request_global_var)


    def getAllTotalArticles(self, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        召回非差 整屋 文章 指南
        @Author: xuzhongjie
        Args:
        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        pagesize = int(root_config["Recall"]["all_total_article_num"].strip())
        # 意图查询召回存在时，取少一点 意图查询召回不存在时，取多一点
        # if len(set(Common.needRunIntentClass) & set(es_recall_params_cls.intentClass)) == 0:
        #     pagesize = int(root_config["Recall"]["all_total_article_num_not_have_intent"].strip())

        fl = self.SEARCH_OP_TOTAL_ARTICLE_HOT_FL

        return esTotalArticleCls.getAllTotalArticle(1, pagesize, fl, es_recall_params_cls, init_resource_cls, init_request_global_var)

    def getVectNoteInfosByIds(self, es_recall_params_cls, init_resource_cls, init_request_global_var, ids):
        """
        通过 note ids 获取 note信息   向量召回的内容ids场景下 使用
        @Author: xuzhongjie
        Args:
            ids: 内容ids
        Returns:

        """
        esNoteCls = esNote(init_resource_cls)

        fl = self.SEARCH_OP_PHOTO_HOT_FL

        return esNoteCls.getVectNoteInfosByIds(ids, es_recall_params_cls, init_resource_cls, init_request_global_var, fl)

    def getVectTotalArticleInfosByIds(self, es_recall_params_cls, init_resource_cls, init_request_global_var, ids):
        """
        通过 ids 获取 total article 信息  向量召回的内容ids场景下 使用
        @Author: xuzhongjie
        Args:
            ids: 内容ids
        Returns:

        """
        esTotalArticleCls = esTotalArticle(init_resource_cls)

        fl = self.SEARCH_OP_TOTAL_ARTICLE_HOT_FL

        return esTotalArticleCls.getVectTotalArticleInfosByIds(ids, es_recall_params_cls, init_resource_cls, init_request_global_var, fl)


EsRecallCls = EsRecall()
# print(EsRecallCls.getGoodIntentClassNote("客厅"))
# print(EsRecallCls.getAllNoteInfos("客厅"))
# print(EsRecallCls.getGoodIntentClassTotalArticle("客厅"))
# print(EsRecallCls.getAllTotalArticle("客厅"))