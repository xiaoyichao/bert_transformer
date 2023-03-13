# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/2 10:34
@Desciption : faiss_map表操作
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

class FaissMap(object):
    @classmethod
    def getAllData(cls, init_resource_cls):
        """
        获取全量faiss map数据
        Returns: list
        """
        sql = '''select id, obj_id, obj_passage_id from hhz_search.faiss_map'''

        dbConnect = init_resource_cls.DB_SEARCH_CONNECT

        content_cursor = dbConnect.cursor()

        content_cursor.execute(sql)

        returnData = []
        for info in content_cursor:
            formatData = {
                "id" : info[0],
                "obj_id" : info[1],
                "obj_passage_id" : info[2],
            }

            returnData.append(formatData)

        content_cursor.close()

        return returnData

