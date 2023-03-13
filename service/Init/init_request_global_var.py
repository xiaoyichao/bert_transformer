# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/5 14:26
@Desciption :  初始化 全局变量
'''
import os
import sys
from common.common_log import Logger, get_log_name
from elasticsearch import Elasticsearch
import configparser
import json
from common.tool import Tool
from common import create_connection

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

dir_name = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(dir_name, "../../config/config.ini")

root_config = configparser.ConfigParser()
root_config.read(config_path)

os.chdir(sys.path[0])
sys.path.append("../../")

class InitRequestGlobalVar(object):
    PARAMS_LOG = {}

    UNIQUE_STR = ""

    def init_log_info(self, params_log):
        """
        初始化参数日志
        Args:
        params_log： 请求参数
        Returns:
        """
        json_str = json.dumps(params_log)
        unique_str = Tool.MD5(json_str)
        params_log["unique_str"] = unique_str
        self.PARAMS_LOG = params_log
        self.UNIQUE_STR = unique_str

# if __name__ == '__main__':
#     InitRequestGlobalVar.init_log_info({"lala":"lalal"})
#     print(InitRequestGlobalVar.PARAMS_LOG)



