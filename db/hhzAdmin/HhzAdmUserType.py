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
@Desciption : hhz_adm_user_type 表操作
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


class HhzAdmUserType(object):
    # 疑似 品牌用户
    AUTH_TAG_TO_BRAND = 105
    # 疑似 设计师用户
    AUTH_TAG_TO_DESIGNER = 104
    # 疑似 商业用户
    AUTH_TAG_SUSPECT_BUSINESS = 103
    @staticmethod
    def getBLevelUserIds(init_resource_cls):
        """
        获取B级用户uids
        Returns: list
        """
        sql = '''select uid from hhz_adm_user_type where quality in (9, 10, 11)'''

        content_cursor = init_resource_cls.DB_ADMIN_CONNECT.cursor()

        content_cursor.execute(sql)

        returnData = set()
        for info in content_cursor:
            returnData.add(info[0])

        content_cursor.close()

        return returnData

