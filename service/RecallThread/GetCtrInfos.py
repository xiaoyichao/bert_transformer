# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-06-10 15:53:50
LastEditTime: 2021-06-10 15:55:18
Description:
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取 交互 rank 的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name
import time
import requests
import socket
import configparser
hostname = socket.gethostname()

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)

class GetCtrInfos(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GetCtrInfos"

    def __init__(self, params_collect_cls, thread_id, queue, init_request_global_var, batch_data_item):
        """
        Args:
            params_collect_cls: 参数收集类
            thread_id: 线程id
            queue: 用于存放召回数据存放
            init_request_global_var: 全局请求参数
        """

        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
        self.init_request_global_var = init_request_global_var
        self.params_collect_cls = params_collect_cls
        self.batch_data_item = batch_data_item

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
            # ctr 预估 时间
            starttime = time.time()
            # ctr_score_dict = requests.post('http://search-ctr.haohaozhu.me/v1/models/search_ctr:predict', data=param,
            #                                timeout=0.5).json()

            ctr_score_dict = {}
            ctr_score = ""
            if len(self.batch_data_item["objIdsUniqueList"]) > 0:
                ctr_score = requests.post(search_opt_common_config["ctr_url"]["url"], data=self.batch_data_item["param"],
                                      timeout=0.5)

                ctr_score_dict = ctr_score.json()
            # ctr_score = requests.post('http://search-ctr.haohaozhu.me/v1/models/search_ctr:predict', data=self.batch_data_item["param"],
            #                           timeout=0.5)

            # print("ctr_score.elapsed time:", ctr_score.elapsed.total_seconds())


            endtime = time.time()
            used_time = endtime - starttime
            log.logger.info(hostname + " ctr 预估 阶段的整体运行时间：{:.10f} s".format(
                used_time) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            ctr_scores = []
            if ctr_score is not None and  "predictions" in ctr_score_dict:
                ctr_scores = ctr_score_dict["predictions"]

            hashObjCtr = {}
            if len(ctr_scores) > 0:
                for obj_unique_id, new_score in zip(self.batch_data_item["objIdsUniqueList"], ctr_scores):
                    hashObjCtr[obj_unique_id] = float(new_score[0])
            else:
                for obj_unique_id in self.batch_data_item["objIdsUniqueList"]:
                    hashObjCtr[obj_unique_id] = 0

            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": hashObjCtr
                }
            )

        except Exception as e:
            err_log.logger.exception(e)
            log.logger.error(self.TYPE + ' 线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" +
                             self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            err_log.logger.error(self.TYPE + ' 线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" +
                                 self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
