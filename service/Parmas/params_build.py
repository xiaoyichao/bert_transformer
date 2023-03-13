# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 10:40
@Desciption :  参数构建相关
'''
from common.tool import Tool
from cache.SearchContent import SearchContent as cSearchContent
import time
import random

class ParamsBuild(object):
    @classmethod
    def getKeyByParams(cls, keyword, params):
        """
        将请求参数加密成字符串 供后续生成缓存key使用
        Args:
            keyword: 搜索词
            params: 参数
        Returns:

        """
        beMd5Str = keyword

        for param in params:
            if type(param) is str:
                beMd5Str += "_" + param
            elif type(param) is list:
                beMd5Str += "_" + "-".join(param)
            elif type(param) is set:
                beMd5Str += "_" + "-".join(param)
            elif type(param) is int:
                beMd5Str += "_" + str(param)

        md5Str = Tool.MD5(beMd5Str)

        return md5Str

    @classmethod
    def addPartSearchKey(cls, init_resource_cls, keyword, search_type):
        """
        添加部分搜索key
        Args:
            keyword: 搜索词
        Returns:

        """
        md5Str = cls.get_part_password_condition(keyword, search_type)
        uniqueKey = Tool.MD5(str(time.time()) + str(random.randint(0, 1000)))

        cSearchContent.addSearchUniqueKey(init_resource_cls, md5Str, uniqueKey)

    @classmethod
    def get_part_password_condition(cls, keyword, search_type):
        return  Tool.MD5(keyword + "search_type:" + str(search_type))

    @classmethod
    def getPartSearchKey(cls, init_resource_cls, keyword, search_type):
        """
        获取部分搜索key
        Args:
            keyword: 搜索词
        Returns:

        """
        md5Str = cls.get_part_password_condition(keyword, search_type)

        searchUniqueKey = cSearchContent.getSearchUniqueKey(init_resource_cls, md5Str)
        if searchUniqueKey is None:
            searchUniqueKey = ""
        else:
            searchUniqueKey = searchUniqueKey.decode()

        return searchUniqueKey

    @classmethod
    def getSearchKey(cls, init_resource_cls, keyword, search_type, params):
        """
        获取完整搜索key
        Args:
            keyword: 搜索词
            params: 参数
        Returns:

        """
        partSearchKey = cls.getPartSearchKey(init_resource_cls, keyword, search_type)

        if len(partSearchKey) > 0:
            searchKey = Tool.MD5(cls.getKeyByParams(keyword, params) + "_" + partSearchKey)
        else:
            searchKey = cls.getKeyByParams(keyword, params)

        return searchKey

    @classmethod
    def buildParamsToBuild(cls, searchType, isOwner, contentTypes, ab_test, uid):
        """
        构建 格式化搜索条件 用于生成搜索key
        Args:
            keyword: 搜索词
            searchType: 搜索类型
            isOwner: 是否只看住友发布
            searchType: 内容类型
        Returns:

        """
        paramsToBuildKey = []
        # 排序方式
        paramsToBuildKey.append(searchType)
        # 是否只看住友发布
        paramsToBuildKey.append(isOwner)
        # 内容类型
        paramsToBuildKey.append(contentTypes)
        # ab 实验
        paramsToBuildKey.append(ab_test)
        # 用户 uid
        paramsToBuildKey.append(uid)

        return paramsToBuildKey

    @classmethod
    def buildSearchKey(cls, init_resource_cls, keyword, search_type, is_owner, content_types, ab_test, uid = 0):
        """
        构建搜索词缓存key
        Args:
            keyword: 搜索词
            searchType: 搜索类型
            is_owner: 是否只看屋主
            content_types: 内容类型
            uid : 当前访问用户uid
        Returns:

        """
        # 缓存中拿数据
        paramsToBuildKey = cls.buildParamsToBuild(search_type, is_owner, content_types, ab_test, uid)

        return cls.getSearchKey(init_resource_cls, keyword, search_type, paramsToBuildKey)


    @classmethod
    def buildEsResultCacheKey(cls, keyword, search_type, is_owner, content_types, ab_test = 1):
        """
        构建搜索词缓存key
        Args:
            keyword: 搜索词
            searchType: 搜索类型
            is_owner: 是否只看屋主
            content_types: 内容类型
        Returns:

        """
        # 缓存中拿数据 设置 uid 为 负数 来生成 key 用于表示存储 es结果的 缓存 key
        paramsToBuildKey = cls.buildParamsToBuild(search_type, is_owner, content_types, ab_test, -10086)

        return cls.getKeyByParams(keyword, paramsToBuildKey)

