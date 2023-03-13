# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao xiaoyichao@haohaozhu.com
Date: 2021-06-10 15:44:34
LastEditTime: 2022-12-07 13:44:21
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取 info 从db中 
'''
from multiprocessing import  Process

import os,queue
from common.common_log import Logger, get_log_name
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from service.Init.InitResource import InitResource
import time

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class dataProcessInsert(Process):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "dataProcessInsert"

    def __init__(self, thread_id, objIdQueue, content_type, obj_ids, threadNum):
        """
        Args:
            thread_id: 线程id
            queue: 用于存放召回数据存放
            esRecallCls: es召回类 对象
        """
        super(dataProcessInsert,self).__init__()
        self.thread_id = thread_id
        self.objIdQueue = objIdQueue  # 任务队列
        self.content_type = content_type  # 任务队列
        self.obj_ids = obj_ids  # 任务队列
        self.threadNum = threadNum  # 任务队列

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

            dataSize = 500
            if self.content_type == 1:
                dataSize = 1000
            elif self.content_type == 2:
                dataSize = 4000
            elif self.content_type == 3:
                dataSize = 1000
            elif self.content_type == 4:
                dataSize = 1000

            objIds = []
            for obj_id in self.obj_ids:
                objIds.append(obj_id)
                if len(objIds) == dataSize:
                    self.objIdQueue.put(objIds)
                    objIds = []

            if len(objIds) > 0:
                self.objIdQueue.put(objIds)

            for i in range(self.threadNum):
                self.objIdQueue.put(None)

        except Exception as e:
            err_log.logger.exception(e)