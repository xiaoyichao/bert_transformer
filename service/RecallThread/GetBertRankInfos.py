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
@Desciption : 获取 Bert rank 的线程
'''
import threading
import os
from common.common_log import Logger, get_log_name
import time
import requests
import orjson
import socket
import configparser
from bert_regress.run_rank import *
hostname = socket.gethostname()

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)

class GetBertRankInfos(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GetBertRankInfos"

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

            headers = {"content-type": "application/json"}

            # 分类排序
            bertRankUrl = search_opt_common_config["ctr_url"]["bert_rank_classify_url"]

            # if self.params_collect_cls.ab_test == 2 :
            #     bertRankUrl = search_opt_common_config["ctr_url"]["bert_rank_classify_pretraining_url"]

            json_response = requests.post(bertRankUrl, data=self.batch_data_item["param"], headers=headers)
            json_data = orjson.loads(json_response.text)
            predictions_tmp = json_data["predictions"]

            predictions = []
            for item in predictions_tmp:
                predictions.append(item["max_scores"])

            endtime = time.time()
            used_time = endtime - starttime
            log.logger.info(hostname + " bert rank 阶段的整体运行时间：{:.10f} s".format(
                used_time) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": predictions,
                    "index" : self.batch_data_item["index"]
                }
            )

        except Exception as e:
            err_log.logger.exception(e)
            log.logger.error(self.TYPE + ' 线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" +
                             self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            err_log.logger.error(self.TYPE + ' 线程错误' + str(e) + " uid:" + self.params_collect_cls.uid + " query:" +
                                 self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
