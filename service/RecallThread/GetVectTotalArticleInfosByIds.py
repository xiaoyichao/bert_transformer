# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 通过内容 ids 获取 total article 信息 的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class GetVectTotalArticleInfosByIds(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GetVectTotalArticleInfosByIds"

    def __init__(self, params_collect_cls, thread_id, queue, esRecallCls, es_recall_params_cls, init_resource_cls, init_request_global_var, ids):
        """
        Args:
            ids: ids
            thread_id: 线程id
            queue: 用于存放召回数据存放
            esRecallCls: es召回类 对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.params_collect_cls = params_collect_cls
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
        self.esRecallCls = esRecallCls  # es召回对象
        self.es_recall_params_cls = es_recall_params_cls
        self.init_resource_cls = init_resource_cls
        self.init_request_global_var = init_request_global_var
        self.ids = ids

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
            import time
            starttime = time.time()
            totalArtilceInfos = self.esRecallCls.getVectTotalArticleInfosByIds(
                self.es_recall_params_cls, self.init_resource_cls, self.init_request_global_var, self.ids)
            endtime = time.time()
            log.logger.debug(
                "es二次内容 getTotalArticleInfosByIds 运行时间：{:.10f} s".format(endtime - starttime))
            # log.logger.debug(allNoteInfos)
            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": totalArtilceInfos
                }
            )
        except Exception as e:
            err_log.logger.error(self.TYPE + ' 线程错误', e)
            log.logger.error(self.TYPE + ' 线程错误', e)
