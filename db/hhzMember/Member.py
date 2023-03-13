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
@Desciption : member 表操作
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


class Member(object):
    # 品牌用户
    AUTH_USER_BRAND = 1
    # 设计师
    AUTH_USER_DESIGNER = 2
    # 个人v认证
    AUTH_USER_V = 3
    # 未认证设计师
    AUTH_USER_UNAUTH_DESIGNER = 4
    # 机构V认证
    AUTH_ORG_V = 5
    # 装修公司
    AUTH_DECO_COMPANY = 6

