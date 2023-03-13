# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/29 21:18
@Desciption :  es参数收集类
'''
from common.tool import Tool
from service.Common import Common

class EsRecallParams():
    # 搜索词
    keyword = ""

    # 切词结果
    splitWords = []

    # 同义词结果
    synonym_map = {}

    # 同义词 核心词
    core_synonym_words = {}

    # 同义词 限定词
    limit_synonym_words = {}

    # 是否只看住友发布
    is_owner = 0

    # 内容类型
    content_types_set = set()

    # 搜索过滤标签
    search_filter_tags = []

    # 意图分类
    intentClass = []

    # 词权重
    terms_weight = {}

    # 核心词
    core_words = {}

    # 限定词
    limit_words = {}

    # ab_test
    ab_test = 1

    # 是否使用昵称召回
    is_use_nick_recall = 0

    # 用户所在地区
    user_area = ""

    # 是否使用地区设计师策略
    is_use_area_designer = 0

