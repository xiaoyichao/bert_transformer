# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-05-11 11:39:48
LastEditTime: 2021-05-11 11:39:49
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/2 10:34
@Desciption : hhz_adm_search_intervention_keyword 表操作
'''
import os
import sys
import configparser
os.chdir(sys.path[0])
sys.path.append("../../")

dir_name = os.path.abspath(os.path.dirname(__file__))
es_config = configparser.ConfigParser()
config_path = os.path.join(dir_name, "../../config/config.ini")

root_config = configparser.ConfigParser()
root_config.read(config_path)


class SearchInterventionKeyword(object):
    STATUS_NORMAL = 1
    # 默认排序
    SORT_TYPE_DEFAULT = 0
    # 最新排序
    SORT_TYPE_NEWEST = 1
    # 删除操作
    OPERATE_TYPE_HIDE = 0
    # 插入操作
    OPERATE_TYPE_NORMAL = 1

    @classmethod
    def getInterventionByRelatedId(cls, init_resource_cls, relatedId, sortType):
        """
        根据 主表关联id 和 排序方式 获取 干预记录
        Returns: list
        """
        strSql = ["obj_id", "operate_type", "intervention_position"]
        sql = '''select %s from search_intervention_keyword where related_id = %s and search_status IN (%s) and sort_type = %s''' % (",".join(strSql), relatedId, cls.STATUS_NORMAL, sortType)

        content_cursor = init_resource_cls.DB_SEARCH_CONNECT.cursor()

        content_cursor.execute(sql)

        returnData = []
        for info in content_cursor:
            formatData = {}
            for strSqlItem, infoItem in zip(strSql, info):
                formatData[strSqlItem] = infoItem
            returnData.append(formatData)

        content_cursor.close()
        return returnData

# if __name__ == '__main__':
#     InitResource.initResource()
#     print(SearchInterventionKeyword.getInterventionByRelatedId(9, SearchInterventionKeyword.SORT_TYPE_DEFAULT))
