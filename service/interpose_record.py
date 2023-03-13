# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/30 16:01
@Desciption : 搜索干预
'''
from db.hhzSearch.SearchIntervention import SearchIntervention
from db.hhzSearch.SearchInterventionKeyword import SearchInterventionKeyword
from service.Common import Common

class InterposeRecord(object):
    @classmethod
    def insertInterposeRecord(cls, init_resource_cls, query, sortTypeStr, objIds):
        """
        将 干预记录 插入 内容ids
        Args:
            query: 搜索词
            sortType:  搜索方式
            objIds: 内容ids

        Returns:

        """
        allAddObjIdsSet, allDeleteObjIdsSet, allRecords = cls.getInterposeRecord(init_resource_cls, query, sortTypeStr)

        objIdsTmp = []
        # 过滤内容ids
        for objId in objIds:
            if objId in allDeleteObjIdsSet or objId in allAddObjIdsSet:
                continue
            objIdsTmp.append(objId)

        # 根据位置信息插入内容列表
        for record in allRecords:
            objIdsTmp = Common.insertContent(record["obj_id"], objIdsTmp, record["position"] - 1)

        return objIdsTmp

    @classmethod
    def getInterposeRecord(cls, init_resource_cls, query, sortTypeStr):
        """
        获取 特定搜索词下 特定搜索方式 的 干预记录
        Args:
            query: 搜索词
            sortTypeStr:  搜索方式

        Returns: list(所有待添加的内容ids, 所有待删除的内容ids, 所有格式化的待添加内容ids包含位置信息)

        """
        # 获取关键词 主表id
        searchInterventionId = SearchIntervention.getIdByKeyword(init_resource_cls, query)
        # 所有需要插入的内容id
        allAddObjIdsSet = set()
        # 所有需要删除的内容id
        allDeleteObjIdsSet = set()
        # 所有需要插入的内容id 及其 位置
        allRecords = []

        if searchInterventionId:
            sortType = SearchInterventionKeyword.SORT_TYPE_NEWEST
            if sortTypeStr == Common.SEARCH_TYPE_HOT_STR:
                sortType = SearchInterventionKeyword.SORT_TYPE_DEFAULT

            # 根据主表id 获取该关键词 下 特定排序下的 干预内容
            interventionInfos = SearchInterventionKeyword.getInterventionByRelatedId(init_resource_cls, searchInterventionId, sortType)

            for interventionInfo in interventionInfos:
                # 插入操作
                if interventionInfo["operate_type"] == SearchInterventionKeyword.OPERATE_TYPE_NORMAL:
                    formatRecordData = {
                        "obj_id": interventionInfo["obj_id"],
                        "position": interventionInfo["intervention_position"],
                    }
                    allRecords.append(formatRecordData)
                    allAddObjIdsSet.add(interventionInfo["obj_id"])
                # 删除操作
                elif interventionInfo.operate_type == SearchInterventionKeyword.OPERATE_TYPE_HIDE:
                    allDeleteObjIdsSet.add(interventionInfo["obj_id"])

        # 按位置正向排序
        if len(allRecords) > 0:
            allRecords.sort(key=lambda i: i['position'])

        return [allAddObjIdsSet, allDeleteObjIdsSet, allRecords]
