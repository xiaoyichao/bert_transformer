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
@Desciption : hhz_adm_search_intervention 表操作
'''
import os
import sys
import configparser
import pymysql
os.chdir(sys.path[0])
sys.path.append("../../")

dir_name = os.path.abspath(os.path.dirname(__file__))
es_config = configparser.ConfigParser()
config_path = os.path.join(dir_name, "../../config/config.ini")

root_config = configparser.ConfigParser()
root_config.read(config_path)


class SearchIntervention(object):
    STATUS_NORMAL = 1

    @classmethod
    def getIdByKeyword(cls, init_resource_cls, keyword):
        """
        根据关键词 获取干预id
        Returns: list
        """
        sql = '''select id from search_intervention where keyword = "%s" and search_status IN (%s)''' % (pymysql.escape_string(keyword), cls.STATUS_NORMAL)

        content_cursor = init_resource_cls.DB_SEARCH_CONNECT.cursor()

        content_cursor.execute(sql)

        id = 0
        for info in content_cursor:
            id = info[0]

        content_cursor.close()

        return id

# if __name__ == '__main__':
#     InitResource.initResource()
#     print(SearchIntervention.getIdByKeyword("客厅"))
