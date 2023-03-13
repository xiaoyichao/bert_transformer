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
@Desciption : hhz tag 表操作
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


class HhzTag(object):
    @staticmethod
    def get_all_admin_tags(init_resource_cls):
        """
        获取所有 tag
        Returns: list
        """
        sqlStrList = ["id", "tag", "tag_alias", "parent_id"]

        sql = "select " + ",".join(sqlStrList) + " from tag where tag_status = 1 order by deep asc"

        content_cursor = init_resource_cls.DB_HHZ_TAG_CONNECT.cursor()

        content_cursor.execute(sql)

        returnData = []
        for info in content_cursor:
            formatData = {}
            for index, sqlStr in enumerate(sqlStrList):
                formatData[sqlStr] = info[index]
            returnData.append(formatData)

        content_cursor.close()

        return returnData

