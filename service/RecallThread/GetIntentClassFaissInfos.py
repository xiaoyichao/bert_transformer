# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取 意图分类下的faiss数据 的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name
import configparser

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


class GetIntentClassFaissInfos(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "IntentClassFaissInfos"

    def __init__(self, queryVec, intentClass, thread_id, queue, searchAnnCls):
        """
        Args:
            queryVec: 搜索词向量数据
            intentClass: 搜索意图
            thread_id: 线程id
            queue: 用于存放召回数据存放
            searchAnnCls: faiss召回类 对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.queryVec = queryVec
        self.intentClass = intentClass
        self.thread_id = thread_id
        self.queue = queue
        self.searchAnnCls = searchAnnCls

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
            _, faissMapIds = self.searchAnnCls.search(self.queryVec, int(root_config["Recall"]["intent_faiss_num"].strip()))
            # log.logger.debug(faissMapIds)
            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": faissMapIds
                }
            )
        except Exception as e:
            err_log.logger.exception(e)
            err_log.logger.error(self.TYPE + ' 线程错误', e)
            log.logger.error(self.TYPE + ' 线程错误', e)

if __name__ == '__main__':
    from ann_engine.faiss_engine4recall import SearchANN
    from queue import Queue

    sentence_list = ["客厅"]
    queryVec = ["vec"]
    dataQueue = Queue()
    searchAnn = SearchANN()
    cls = GetIntentClassFaissInfos(queryVec,  ["装修知识"], GetIntentClassFaissInfos.TYPE,
                                   dataQueue,  searchAnn)
    cls.start()
    cls.join()
    print(dataQueue.get(block=False))
