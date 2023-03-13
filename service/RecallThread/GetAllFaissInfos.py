# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取 全量的faiss数据 的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name
from service.Common import Common
import configparser
import time
import grpc

from rpc_server.api import vec_recall_pb2_grpc, vec_recall_pb2
from google.protobuf.json_format import MessageToDict

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

all_faiss_num = int(root_config["Recall"]["all_faiss_num"].strip())

common_config = configparser.ConfigParser()
common_config.read("/home/resource_config/search_opt/common.ini")

vec_recall_url = common_config["vec_recall"]["rpc_url"]
vec_recall_port = common_config["vec_recall"]["rpc_port"]

_HOST = vec_recall_url
_PORT = vec_recall_port

class GetAllFaissInfos(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "AllFaissInfos"

    def __init__(self, query, thread_id, queue, ab_test, unique_str, uid):
        """
        Args:
            queryVec: 搜索词向量数据
            thread_id: 线程id
            queue: 用于存放召回数据存放
            searchAnnCls: faiss召回类 对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.query = query
        self.thread_id = thread_id
        self.queue = queue
        self.ab_test = ab_test
        self.unique_str = unique_str
        self.uid = uid

    def run(self):
        '''
        线程在调用过程中就会调用对应的run方法
        :return:
        '''
        self.recall()

    def recall(self):
        """
        从faiss召回数据并将数据存入queue中
        Returns:

        """
        try:
            starttime = time.time()

            num = int(root_config["Recall"]["all_faiss_num"].strip())

            with grpc.insecure_channel("{0}:{1}".format(_HOST, _PORT)) as channel:
                client = vec_recall_pb2_grpc.VecRecallServiceStub(channel=channel)
                response = client.VecRecall(vec_recall_pb2.VecReq(keyword=str(self.query), unique_str=str(self.unique_str), uid=int(self.uid), num=int(num)))

            # 内容ids 拆分 note 和 非note
            dataMap = {
                "note": [],
                "not_note": [],
                "obj_ids": [],
            }

            res = MessageToDict(response)

            if 'code' in res and res['code'] == 1:
                similars = []
                obj_ids = []

                if "data" in res and "similars" in res["data"] and "objIds" in res["data"]:
                    similars = res["data"]["similars"]

                    obj_ids = res["data"]["objIds"]

                if len(obj_ids) > 0:
                    dataMap["obj_ids"] = obj_ids

                    for objId, similar in zip(obj_ids, similars):
                        if len(objId) == 0:
                            continue

                        if Common.is_obj_id_note(objId):
                            dataMap["note"].append(objId)
                        else:
                            dataMap["not_note"].append(objId)

            self.queue.put(
                {
                    "type" : self.TYPE,
                    "data" : dataMap
                }
            )


            endtime = time.time()
            log.logger.info("GetAllFaissInfos 获取 向量召回 运行时间：{:.10f} s".format(
                endtime - starttime) + " uid:" + str(self.uid) + " query:" + self.query + " unique_str:" + self.unique_str)


        except Exception as e:
            err_log.logger.exception(e)
            log.logger.error(self.TYPE + ' 线程错误', e)
            err_log.logger.error(self.TYPE + ' 线程错误', e)


if __name__ == '__main__':
    from ann_engine.faiss_engine4recall import SearchANN
    from queue import Queue

    sentence_list = ["客厅"]
    dataQueue = Queue()
    searchAnn = SearchANN()
    cls = GetAllFaissInfos("客厅", GetAllFaissInfos.TYPE, dataQueue, 1, 1, 1)
    cls.start()
    cls.join()
    print(dataQueue.get(block=False))
