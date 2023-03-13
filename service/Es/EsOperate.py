# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/6 22:06
@Desciption : es 操作服务 添加/更新数据 删除数据
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")

from es.Note import Note
from es.TotalArticle import TotalArticle
from service.Init.InitResource import InitResource
import requests
from service.Common import Common

class EsOperate(object):
    # 保存数据 操作标识
    SAVE_DATA_OPERATE = "SaveData"
    # 删除数据 操作标识
    DELETE_DATA_OPERATE = "DeleteData"
    # note 索引 标识
    SEARCH_OP_PHOTO = "SearchOpPhoto"
    # total article 索引 标识
    SEARCH_OP_TOTAL_ARTICLE = "SearchOpTotalArticle"

    PYTHON_DATA_EMPTY = "python_data_empty"

    def SaveData(self, index, dataDict):
        """
        保存数据进es
        Args:
            index:  索引标识
            dataDict:  接收数据

        Returns:

        """
        id = ""
        saveData = {}

        for name, value in dataDict.items():
            if name == "id":
                id = value[0]

            data = value[0]
            # 由于sanci request.form 接收空值数据 会自动过滤key 所以重新定义空值
            if data == self.PYTHON_DATA_EMPTY:
                data = ""

            saveData[name] = data

        init_resource_cls = InitResource()
        init_resource_cls.initResource()

        if index == self.SEARCH_OP_PHOTO:
            # all_keyword
            saveData["all_keyword"] = saveData["remark"] + " " + saveData["title"]
            saveData["all_keyword"] = saveData["all_keyword"].strip()

            split_word_str = ""
            split_word_list = []

            if len(saveData["all_keyword"]) > 0:
                split_word_str, split_word_list = Common.get_split_data(saveData["all_keyword"])
            # split_words_for_ai
            saveData["split_words_for_ai"] = split_word_str
            saveData["split_words_array"] = split_word_list

            rela_wiki_names_split_str = ""
            rela_wiki_names_split_list = []

            saveData["rela_wiki_names"] = saveData["rela_wiki_names"].strip()
            if len(saveData["rela_wiki_names"]) > 0:
                rela_wiki_names_split_str, rela_wiki_names_split_list = Common.get_split_data(saveData["rela_wiki_names"])

            saveData["rela_wiki_names_split"] = rela_wiki_names_split_str
            saveData["rela_wiki_names_split_array"] = rela_wiki_names_split_list

            saveData["split_words_for_ai_test"] = ""

            # 标题
            saveData["split_words_for_ai_title"] = ""
            saveData["split_words_title_array"] = []

            if len(saveData["title"]) > 0:
                saveData["split_words_for_ai_title"], saveData["split_words_title_array"] = Common.get_split_data(saveData["title"], 0, sep = "奰")

            # 描述
            saveData["split_words_for_ai_remark"] = ""
            saveData["split_words_remark_array"] = []

            remarkLen = len(saveData["remark"])
            if remarkLen > 0:
                if remarkLen > 128:
                    remarkLen = 128

                saveData["split_words_for_ai_remark"], saveData["split_words_remark_array"] = Common.get_split_data(saveData["remark"][:remarkLen], 0, sep = "奰")

            noteCls = Note(init_resource_cls)
            noteCls.saveData(id, saveData)

        elif index == self.SEARCH_OP_TOTAL_ARTICLE:
            split_word_str = ""
            split_word_list = []

            saveData["title"] = saveData["title"].strip()
            if len(saveData["title"]) > 0:
                split_word_str, split_word_list = Common.get_split_data(saveData["title"])
            # split_words_for_ai
            saveData["split_words_for_ai"] = split_word_str
            saveData["split_words_array"] = split_word_list

            rela_wiki_names_split_str = ""
            rela_wiki_names_split_list = []

            saveData["rela_wiki_names"] = saveData["rela_wiki_names"].strip()
            if len(saveData["rela_wiki_names"]) > 0:
                rela_wiki_names_split_str, rela_wiki_names_split_list = Common.get_split_data(saveData["rela_wiki_names"])

            saveData["rela_wiki_names_split"] = rela_wiki_names_split_str
            saveData["rela_wiki_names_split_array"] = rela_wiki_names_split_list

            saveData["split_words_for_ai_test"] = ""

            # 标题
            saveData["split_words_for_ai_title"] = ""
            saveData["split_words_title_array"] = []
            if len(saveData["title"]) > 0:
                saveData["split_words_for_ai_title"], saveData["split_words_title_array"] = Common.get_split_data(saveData["title"], 0, sep = "奰")

            # 描述
            saveData["split_words_for_ai_remark"] = ""
            saveData["split_words_remark_array"] = []

            remarkLen = len(saveData["fulltext"])
            if remarkLen > 0:
                if remarkLen > 128:
                    remarkLen = 128

                saveData["split_words_for_ai_remark"], saveData["split_words_remark_array"] = Common.get_split_data(saveData["fulltext"][:remarkLen], 0, sep = "奰")

            totalArticleCls = TotalArticle(init_resource_cls)
            totalArticleCls.saveData(id, saveData)

    def DeleteData(self, index, dataDict):
        """
        删除es数据
        Args:
            index: 索引标识
            dataDict: 接收数据

        Returns:

        """
        id = ""

        for name, value in dataDict.items():
            if name == "id":
                id = value[0]

        init_resource_cls = InitResource()
        init_resource_cls.initResource()

        if len(id) > 0:
            if index == self.SEARCH_OP_PHOTO:
                noteCls = Note(init_resource_cls)
                noteCls.deleteData(id)
            elif index == self.SEARCH_OP_TOTAL_ARTICLE:
                totalArticleCls = TotalArticle(init_resource_cls)
                totalArticleCls.deleteData(id)

