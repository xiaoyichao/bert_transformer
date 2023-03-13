# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/2 10:34
@Desciption : obj_id2index表操作
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

class ObjId2Index(object):
    @classmethod
    def getAllObjId2Index(cls, init_resource_cls):
        """
        获取全量 obj id 和 id 数据
        Returns: list
        """
        sql = '''select id, obj_id from hhz_search.obj_id2index'''

        dbConnect = init_resource_cls.DB_SEARCH_CONNECT

        content_cursor = dbConnect.cursor()

        content_cursor.execute(sql)

        objId2Index = {}
        for info in content_cursor:
            objId2Index[info[1]] = info[0]

        return objId2Index

    @classmethod
    def insertDatas(cls, init_resource_cls, datas):
        if len(datas) > 0:
            insertStr = ''

            for data in datas:
                insertStr += '("'+ data + '"),'

            insertStr = insertStr.strip(",")
            sql = 'insert IGNORE into hhz_search.obj_id2index (`obj_id`) VALUES ' + insertStr

            dbConnect = init_resource_cls.DB_SEARCH_CONNECT

            content_cursor = dbConnect.cursor()
            content_cursor.execute(sql)


if __name__ == '__main__':
    from service.Init.InitResource import InitResource

    InitResourceCls = InitResource()
    InitResourceCls.initDbSearchConnect()

    # print(ObjId2Index.getAllObjId2Index(InitResourceCls))
    print(ObjId2Index.insertDatas(InitResourceCls, ["aaaa", "bbbb"]))

