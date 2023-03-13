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
@Desciption : 数据预处理 为 用户搜索个性化
'''
import threading
import os
import requests
import orjson
from common.common_log import Logger, get_log_name
import time
import configparser


abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)


class DataPrepareForUserProfile(threading.Thread):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "DATA_PREPARE_FOR_USER"

    def __init__(self, sortedObjIds_top, query, user_profile_data, user_click_seq_historys, user_favorite_seq_historys,
                     hashEsDataInfos, query_splited_list, params_collect_cls, init_request_global_var, queue, index):
        """
        Args:
            sortedObjIds_top, 待排序的 内容id
            query, 搜索词
            user_profile_data, 用户分群数据
            user_click_seq_historys, 用户有效点击序列
            user_favorite_seq_historys, 用户收藏序列
            hashEsDataInfos, es数据
            query_splited_list, 分词结果
            params_collect_cls 参数收集对象
        """
        threading.Thread.__init__(self)  # 需要对父类的构造函数进行初始化
        self.sortedObjIds = sortedObjIds_top
        self.query = query
        self.user_profile_data = user_profile_data
        self.user_click_seq_historys = user_click_seq_historys  # 任务队列
        self.user_favorite_seq_historys = user_favorite_seq_historys
        self.hashEsDataInfos = hashEsDataInfos
        self.query_splited_list = query_splited_list
        self.params_collect_cls = params_collect_cls
        self.queue = queue
        self.init_request_global_var = init_request_global_var
        self.index = index

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

            needPostHashEsData = {}
            needInHashEsDataKeys = ["title", "desc", "favorite", "comment", "like", "admin_score", "publish_time", "is_relate_wiki", "split_words_title_array", "split_words_remark_array"]

            for sortedObjId in self.sortedObjIds:
                if sortedObjId in self.hashEsDataInfos:
                    needPostHashEsData[sortedObjId] = {}
                    for needInHashEsDataKey in needInHashEsDataKeys:
                        if needInHashEsDataKey in self.hashEsDataInfos[sortedObjId]:
                            if needInHashEsDataKey == "desc" :
                                needPostHashEsData[sortedObjId][needInHashEsDataKey] = self.hashEsDataInfos[sortedObjId][needInHashEsDataKey][:64]
                            else:
                                needPostHashEsData[sortedObjId][needInHashEsDataKey] = self.hashEsDataInfos[sortedObjId][needInHashEsDataKey]


            post_dict = {
                "sorted_obj_ids": self.sortedObjIds,
                "query": self.query,
                "user_profile_data": self.user_profile_data,
                "user_click_seq_historys": self.user_click_seq_historys,
                "user_favorite_seq_historys": self.user_favorite_seq_historys,
                "hashEsDataInfos": needPostHashEsData,
                "query_splited_list": self.query_splited_list,
                "index": self.index,
                "ab_test": self.params_collect_cls.ab_test,
                "uid": self.params_collect_cls.uid,
                "unique_str": self.init_request_global_var.UNIQUE_STR
            }

            postData = {
                "json_data": orjson.dumps(post_dict)
            }

            headers = {
                'Accept': 'text/plain;charset=utf-8',
                'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
            }

            response = requests.post(
                search_opt_common_config["search_opt"]["prepare_feature_search_url"], headers=headers, data=postData)

            result = orjson.loads(response.content)

            endtime = time.time()

            log.logger.info("构建 特征数据 为 用户个性化搜索 的 运行时间：{:.10f} s".format(
                    endtime - starttime) + " uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query +  " unique_str:" + self.init_request_global_var.UNIQUE_STR)

            self.queue.put(
                {
                    "type": self.TYPE,
                    "data": result["data"]["features"],
                    "index" : self.index
                }
            )

        except Exception as e:
            err_log.logger.exception(e)
            err_log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
            log.logger.error(self.TYPE + ' 线程错误' + str(e) + "uid:" + self.params_collect_cls.uid + " query:" + self.params_collect_cls.query + " unique_str:" + self.init_request_global_var.UNIQUE_STR)
