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
@Desciption : tag package 表操作
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

class TagPackage(object):
    PACKAGE_STATUS_NORMAL = 1

    @classmethod
    def get_search_filter_tags(cls, init_resource_cls):
        """
        获取 搜索过滤 tag
        Returns: list
        """
        sqlStrList = ["tags"]

        sql = "select " + ",".join(sqlStrList) + " from tag_package where description = '搜索过滤' and package_status = " + str(cls.PACKAGE_STATUS_NORMAL)

        tag_cursor = init_resource_cls.DB_HHZ_TAG_CONNECT.cursor()

        tag_cursor.execute(sql)

        returnData = []
        for info in tag_cursor:
            if info is not None and info[0] is not None and len(info[0]) > 0:
                tags = info[0].split(",")

                if len(tags) > 0:
                    returnData.extend(tags)

        tag_cursor.close()

        return returnData

# if __name__ == '__main__':
#     InitResource.initResource()
#     print(TagPackage.get_search_filter_tags())

