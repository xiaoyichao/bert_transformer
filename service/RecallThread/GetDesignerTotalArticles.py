# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-06-10 16:07:54
LastEditTime: 2021-06-10 16:08:34
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取优质意图 整屋，文章，指南 信息的线程
'''

import threading
import os
import sys
import time
os.chdir(sys.path[0])
sys.path.append("../")
from service.Es.EsRecall import EsRecall
from common.common_log import Logger, get_log_name


abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class GetDesignerTotalArticles(threading.Thread):
    '''
        线程类，注意需要继承线程类Thread
        '''
    # 标记线程类
    TYPE = "GetDesignerTotalArticles"

    def __init__(self, params_collect_cls, thread_id, queue, esRecallCls,  es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        Args:
            thread_id: 线程id
            queue: 用于存放召回数据存放
            esRecallCls: es召回类 对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
        self.esRecallCls = esRecallCls  # es召回对象
        self.es_recall_params_cls = es_recall_params_cls
        self.init_resource_cls = init_resource_cls
        self.init_request_global_var = init_request_global_var
        self.params_collect_cls = params_collect_cls

    def run(self):
        '''
        线程在调用过程中就会调用对应的run方法
        :return:
        '''
        self.recall()

    def recall(self):
        """
        从es召回数据并将数据存入queue中
        Returns:

        """
        try:
            starttime = time.time()
            DesignerTotalArticles = self.esRecallCls.getDesignerTotalArticles(self.es_recall_params_cls, self.init_resource_cls, self.init_request_global_var)
            endtime = time.time()
            log.logger.info("GetDesignerTotalArticles 运行时间：{:.10f} s".format(
                    endtime - starttime) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type" : self.TYPE,
                    "data" : DesignerTotalArticles
                }
            )
        except Exception as e:
            err_log.logger.exception(e)
            log.logger.error(self.TYPE + ' 采集线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            err_log.logger.error(self.TYPE + ' 采集线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

if __name__ == '__main__':
    from queue import Queue
    dataQueue = Queue()
    esCls = EsRecall()
    cls = GetDesignerTotalArticles(GetDesignerTotalArticles.TYPE,
                                          dataQueue,  esCls)
    cls.start()
    cls.join()
    print(dataQueue.get(block=False))