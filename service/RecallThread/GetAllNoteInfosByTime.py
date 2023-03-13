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
@Desciption : 全量非差note信息 带时间字段 用于排序的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name
import time

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)


class GetAllNoteInfosByTime(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GetAllNoteInfosByTime"

    def __init__(self, params_collect_cls, thread_id, queue, esRecallCls, es_recall_params_cls, init_resource_cls, init_request_global_var):
        """
        Args:
            keyword: 搜索词
            splitWords: 搜索词分词结果
            SynonymWords: 同义词
            is_owner: 是否是住友发布
            content_types_set: 内容类型
            search_filter_tags: 搜索过滤标签
            thread_id: 线程id
            queue: 用于存放召回数据存放
            esRecallCls: es召回类 对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.es_recall_params_cls = es_recall_params_cls
        self.init_resource_cls = init_resource_cls
        self.init_request_global_var = init_request_global_var
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
        self.esRecallCls = esRecallCls  # es召回对象
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
            allNoteInfos = self.esRecallCls.getAllNoteInfosByTime(self.es_recall_params_cls, self.init_resource_cls, self.init_request_global_var)
            endtime = time.time()

            log.logger.info("GetAllNoteInfosByTime 运行时间：{:.10f} s".format(
                    endtime - starttime) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type" : self.TYPE,
                    "data" : allNoteInfos
                }
            )
        except Exception as e:
            err_log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)