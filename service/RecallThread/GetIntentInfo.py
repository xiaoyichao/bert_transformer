# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-06-10 16:14:56
LastEditTime: 2021-06-10 16:15:27
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取语义向量
'''
import threading
import os
from common.common_log import Logger, get_log_name
import time
from service.Common import Common

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)


class GetIntentInfo(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GET_INTENT_INFO"

    def __init__(self, queryVec, params_collect_cls, thread_id, queue, init_resource_cls, init_request_global_var):
        """
        Args:
            params_collect_cls: 参数收集类
            init_resource_cls: 资源类
            init_request_global_var: 全局变量类
            thread_id: 线程id
            queue: 用于存放召回数据存放
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.init_resource_cls = init_resource_cls
        self.init_request_global_var = init_request_global_var
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
        self.params_collect_cls = params_collect_cls
        self.queryVec = queryVec

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
            # 获取搜索词意图
            starttime = time.time()
            intentClass = Common.get_intent_info(self.init_resource_cls, self.params_collect_cls.query, self.queryVec)
            endtime = time.time()

            log.logger.info("获取搜索词意图 的 运行时间：{:.10f} s".format(
                    endtime - starttime) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " 意图分类：" + ",".join(intentClass)  + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": intentClass
                }
            )
        except Exception as e:
            err_log.logger.exception(e)
            err_log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
