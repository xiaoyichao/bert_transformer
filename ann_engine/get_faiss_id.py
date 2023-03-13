# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-04-29 17:14:01
LastEditTime: 2021-10-06 20:41:28
Description: faiss中使用的id依靠mysql维护
'''

import os
import sys
import time
import configparser
os.chdir(sys.path[0])
sys.path.append("../")
from common import create_connection
from common.common_log import Logger, get_log_name


abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

dir_name = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(dir_name, "../config/config.ini")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

root_config = configparser.ConfigParser()
root_config.read(config_path)


class FaissId(object):
    '''
    @Author: xiaoyichao
    @param {*} self
    @Description: 关于faiss使用的id（faiss_map表中的id）的类
    '''
    @classmethod
    def get_content_type(cls, obj_id):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 根据id判断文章类型
            第 9  位
            0 为 note
            1为 整屋
            2为 指南
            4为 回答的note
            5为 文章
            {content_type:1是文章，2是note，3是整屋，4是指南}
        '''
        id_pos_9 = int(list(str(obj_id))[8])
        if id_pos_9 == 0 or id_pos_9 == 4:  # 0,4 为 note
            content_type = 2
        elif id_pos_9 == 1:  # 1为 整屋
            content_type = 3
        elif id_pos_9 == 2:  # 2为 指南
            content_type = 4
        elif id_pos_9 == 5:  # 5为 文章
            content_type = 1
        else:
            content_type = None  # 异常数据
            log.logger.error("增加数据的时候，发现异常数据,obj_id:%s" % obj_id)
            err_log.logger.error("增加数据的时候，发现异常数据,obj_id:%s" % obj_id)
        return content_type

    def write_ids2sql(self, obj_passage_ids: list, content_type=None):
        '''
        @Author: xiaoyichao
        @param {content_type:1是文章，2是note，3是整屋，4是指南}
        @Description: 将obj_passdge_id信息写入mysql，生成int(id),用于给faiss作为id使用
        '''
        add_values = []
        add_time = int(time.time())
        edit_time = add_time
        for obj_passage_id in obj_passage_ids:
            obj_id = obj_passage_id.split("_")[0]
            if content_type is None:  # 如果没有指定content_type，则根据obj_id判断content_type
                content_type = self.get_content_type(obj_id)
            if content_type is not None:
                add_values.append(
                    (obj_id, obj_passage_id, add_time, edit_time, content_type))
            else:
                log.logger.error("content_type is None的数据,obj_id:%s" % obj_id)
                err_log.logger.error("content_type is None的数据,obj_id:%s" % obj_id)

        log.logger.info("解析完成obj_id 相关信息，待写入mysql")
        sql = '''INSERT INTO  hhz_search.faiss_map(obj_id, obj_passage_id, add_time, edit_time, content_type) VALUES(%s,%s,%s,%s,%s) '''
        db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')
        cursor = db.cursor()
        cursor.executemany(sql, add_values)
        db.commit()
        cursor.close()
        db.close()
        log.logger.info("obj_id 相关信息，已经写入mysql")

    def get_all_faiss_map_ids(self):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 读取mysql中的id信息，以两个字典的方式写到内存中
        '''
        sql = ''' SELECT id, obj_id, obj_passage_id from hhz_search.faiss_map'''
        db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')
        cursor = db.cursor()
        cursor.execute(sql)
        infos = cursor.fetchall()
        objid_id_dict = {}
        # objid_id_dict_keys = set()
        objpassageid_id_dict = {}
        ids = []
        for i, info in enumerate(infos):
            id, obj_id, obj_passage_id = info[0], info[1], info[2]
            objpassageid_id_dict[obj_passage_id] = id
            ids.append(id)
            ''' objid 可能对应不同的id,因为有的内容有多个段落（512 token）'''
            objid_id_dict.setdefault(obj_id, []).append(id)
            # 一句话实现下边的功能
            # if obj_id not in objid_id_dict_keys:
            #     objid_id_dict[obj_id] = id
            #     objid_id_dict_keys.add(obj_id)
            # else:
            #     ids = []
            #     ids.append(objid_id_dict[obj_id])
            #     objid_id_dict[obj_id] = ids
            if i % 10000 == 0:
                log.logger.info("从hhz_search.faiss_map中读取了%s条数据" % i)
        cursor.close()
        db.close()
        log.logger.info("id和obj_id, obj_passage_id的对应关系已经存储到字典")
        return objid_id_dict, objpassageid_id_dict, ids

    def get_faiss_map_id4obj_id(self, obj_id):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 根据obj_id，读取mysql中的id信息
        '''
        sql = ''' SELECT id from hhz_search.faiss_map WHERE obj_id=\'%s\' ''' % obj_id
        db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')
        log.logger.debug("get_faiss_map_id4obj_id_sql%s" % sql)
        cursor = db.cursor()
        cursor.execute(sql)
        infos = cursor.fetchall()
        faiss_map_id_list = []
        # objid_id_dict = {}
        # objpassageid_id_dict = {}
        # ids = []
        for info in infos:
            faiss_map_id = info[0]
            faiss_map_id_list.append(faiss_map_id)
            ''' objid 可能对应不同的id,因为有的内容有多个段落（512 token）'''
        cursor.close()
        db.close()
        return faiss_map_id_list

    def get_faiss_map_id4obj_pass_id(self, obj_pass_id):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 根据obj_pass_id，读取mysql中的id信息
        '''
        sql = ''' SELECT id from hhz_search.faiss_map WHERE obj_passage_id=\'%s\' ''' % obj_pass_id
        db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')
        log.logger.debug("get_faiss_map_id4obj_pass_id_sql%s" % sql)
        cursor = db.cursor()
        cursor.execute(sql)
        infos = cursor.fetchall()
        # faiss_map_id_list = []
        # objid_id_dict = {}
        # objpassageid_id_dict = {}
        # ids = []
        # for info in infos:
        #     faiss_map_id = infos[0]
        #     faiss_map_id_list.append(faiss_map_id)
        #     ''' objid 可能对应不同的id,因为有的内容有多个段落（512 token）'''
        cursor.close()
        db.close()
        return infos[0][0]

    def del_faiss_map(self):
        '''
        @Author: xiaoyichao
        @param {}
        @Description: 删除faiss_map中的数据
        '''
        sql = '''DELETE FROM  hhz_search.faiss_map '''
        db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        cursor.close()
        db.close()
        log.logger.info("清空faiss_map中的数据 done")

    def del_faiss_map_obj_id(self, obj_ids_list: list):
        '''
        @Author: xiaoyichao
        @param {}
        @Description: 删除faiss_map中某些数据
        '''
        if len(obj_ids_list) > 0:
            sql = '''DELETE FROM  hhz_search.faiss_map WHERE obj_id in (%s)''' % str(
                obj_ids_list).replace("[", "").replace("]", "")
            log.logger.debug("del_faiss_map_obj_id_sql%s" % str(sql))
            db = create_connection.MySQLCli.connect(
                root_config["FileLocation"]['db_search'], 'db_search')
            cursor = db.cursor()
            cursor.execute(sql)
            db.commit()
            cursor.close()
            db.close()
            log.logger.info("删除faiss_map中 过期的数据 done")
        else:
            log.logger.info("len(obj_ids_list)==0,没有要删除的faiss_map中的数据 done")


'''测试代码'''
if __name__ == '__main__':
    faissid = FaissId()
    # faissid.del_faiss_map()
    # faissid.write_ids2sql(["qweq_1","sdad_1"])
    # objid_id_dict, objpassageid_id_dict, ids = faissid.get_id()
    # print(objid_id_dict, objpassageid_id_dict, ids)
    # faissid.del_faiss_map_obj_id(["00000f6050009msx", "000000t05000096o"])
    id = faissid.get_faiss_map_id4obj_pass_id("000000t05000096o_1")
    print(id)
