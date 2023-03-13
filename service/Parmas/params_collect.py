# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/29 21:18
@Desciption :  参数收集类
'''
from common.tool import Tool
from service.Common import Common
import time
import random
from service.Init.InitService import InitService

class ParamsCollect():
    # 搜索词
    query = ""
    # uid
    uid = 0
    # user_agent
    user_agent = ""
    # 处理过后 的 搜索词
    query_trans = ""
    # page
    page = 1
    # pagesize
    pagesize = 20
    # 搜索缓存key
    searchCacheKey = ""
    # 搜索类型
    searchType = Common.SEARCH_TYPE_HOT
    # 是否只看住友 即 只看普通用户
    is_owner = 0
    # 内容类型
    content_types = ""
    # 参数日志
    params_log = None
    # a b 实验
    ab_test = 1
    # 是否使用昵称 召回策略
    is_use_nick_recall = 1

    # 是否使用地区设计师策略
    is_use_area_designer = 0

    # 用户所在地区
    user_area = ""

    def collect(self, request):
        params_log = {}

        # 用于生成随机数
        params_log["current_time"] = time.time()
        params_log["random"] = random.randint(0, 10000)

        # 搜索词
        query = request.form.get("query")
        if query is None:
            query = ""

        params_log["query"] = query
        self.query = query

        # uid
        uid = str(request.form.get("uid"))
        params_log["uid"] = uid
        self.uid = uid

        # ua
        user_agent = str(request.form.get("user_agent"))
        params_log["user_agent"] = user_agent
        self.user_agent = user_agent

        # query = "客厅"
        tool = Tool()
        query_trans = tool.T2S(query)
        params_log["query_trans"] = query_trans
        self.query_trans = query_trans

        # page 不传默认第一页
        page = request.form.get("page")
        params_log["page"] = page
        if page is None:
            page = 1
        page = int(page)

        self.page = page

        # pagesize条数 不传默认20条
        pagesize = request.form.get("pagesize")
        params_log["pagesize"] = pagesize

        if pagesize is None:
            pagesize = Common.PAGESIZE
        pagesize = int(pagesize)

        self.pagesize = pagesize

        # 缓存 search_key
        searchCacheKey = request.form.get("search_key")
        params_log["searchCacheKey"] = searchCacheKey
        if searchCacheKey is None:
            searchCacheKey = ""

        self.searchCacheKey = searchCacheKey

        # 搜索类型 热度 还是 时间
        searchType = request.form.get("search_type")
        params_log["searchType"] = searchType
        if searchType is None:
            searchType = Common.SEARCH_TYPE_HOT
        else:
            searchType = int(searchType)

        self.searchType = searchType

        # 只看住友发布
        is_owner = request.form.get("is_owner")
        params_log["is_owner"] = is_owner
        if is_owner is None:
            is_owner = 0
        else:
            is_owner = int(is_owner)

        self.is_owner = is_owner

        # 内容筛选
        content_types = request.form.get("content_types")
        params_log["content_types"] = content_types
        if content_types is None:
            content_types = ""

        self.content_types = content_types

        # 参数日志
        self.params_log = params_log

        # ab 实验
        ab_test = request.form.get("ab_test")
        if ab_test is None:
            ab_test = 1
        self.ab_test = int(ab_test)

        # if self.ab_test == 2 or self.ab_test == 3:
        #     self.ab_test = 1
        # ab_test = 4
        # self.ab_test = 4

        # 用户地区
        user_area = request.form.get("user_area")
        if user_area is None:
            user_area = ""
        self.user_area = user_area

        # 是否使用地区设计师召回策略
        if user_area in InitService.hash_city_designer_ratio:
            self.is_use_area_designer = 1
        # if self.ab_test == 1:
        #     self.is_use_area_designer = 0
        # else:
        #     self.is_use_area_designer = 1