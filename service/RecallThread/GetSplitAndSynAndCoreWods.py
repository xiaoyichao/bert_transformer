# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-06-10 16:14:56
LastEditTime: 2021-07-19 19:05:29
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
from urllib3 import exceptions
import requests
import json
import configparser

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

common_config = configparser.ConfigParser()
common_config.read("/home/resource_config/search_opt/common.ini")
bert_server_url = common_config["bert_server_url"]["url"]

class GetSplitAndSynAndCoreWods(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "GetSplitAndSynWods"

    def __init__(self, params_collect_cls, thread_id, queue, init_resource_cls, init_request_global_var):
        """
        Args:
            params_collect_cls: 参数收集类
            thread_id: 线程id
            init_resource_cls: 初始化资源
            init_request_global_var: 初始化 全局变量
            queue: 用于存放召回数据存放
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.init_resource_cls = init_resource_cls
        self.init_request_global_var = init_request_global_var
        self.thread_id = thread_id
        self.queue = queue  # 任务队列
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
            SplitAndSynAndCoreWods = {
                "split_words": [],
                "synonym_words": [],
                "terms_weight": {}
            }

            headers = {
                'Accept': 'text/plain;charset=utf-8',
                'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
            }
            data = {"original_query": self.params_collect_cls.query,
                    "unique_str": self.init_request_global_var.UNIQUE_STR}
            response = requests.post(
                bert_server_url + '/Cut_word_Term_weight', headers=headers, data=data)
            result = eval(response.content)

            if 'code' in result and result['code'] == 1:
                splitWords = result["data"]["query_list"]

                log.logger.info(
                    "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " 分词结果为：" + ",".join(
                        splitWords) + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

                SplitAndSynAndCoreWods["split_words"] = splitWords

                # 同义词结果
                synonym_map = result["data"]["synonym_map"]

                log.logger.info(
                    "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " 同义词结果为：" + json.dumps(synonym_map, ensure_ascii=False) + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

                SplitAndSynAndCoreWods["synonym_map"] = synonym_map

                # 词权重
                SplitAndSynAndCoreWods["terms_weight"] = result["data"]["term_weight_result"]

                log.logger.info(
                    "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " 词权重结果为：" + json.dumps(SplitAndSynAndCoreWods["terms_weight"], ensure_ascii=False) + " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": SplitAndSynAndCoreWods
                }
            )
        except Exception as e:
            err_log.logger.exception(e)
            err_log.logger.error(self.TYPE + ' 线程错误' + str(
                e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            log.logger.error(self.TYPE + ' 线程错误' + str(
                e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)


if __name__ == '__main__':
    test = [1, 2, 3]
    test = set(test)
    import random
    print(random.sample(test, 2))
