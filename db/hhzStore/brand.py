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
@Desciption : store brand 表操作
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


class Brand(object):
    # 状态为启用
    STATUS_USING = 1

    TABLE_NAME = "brand"

    @classmethod
    def getAllBrand(cls, init_resource_cls):
        """
        获取 所有 品牌
        Returns: list
        """
        sql = "select brand_name, en_brand_name, brand_alias  from "+ cls.TABLE_NAME + "  where status in (" + str(cls.STATUS_USING)  + ")"

        content_cursor = init_resource_cls.DB_STORE_CONNECT.cursor()

        content_cursor.execute(sql)

        returnData = []
        for info in content_cursor:
            formatData = {
                "brand_name" : info[0],
                "en_brand_name" : info[1],
                "brand_alias" : info[2],
            }
            returnData.append(formatData)

        content_cursor.close()

        return returnData

