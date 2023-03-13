# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-06-29 13:47:38
LastEditTime: 2021-07-26 14:34:27
Description: 设计师流量池的数据
'''
import random
import redis
import configparser
import readline  # 防止方向键乱码
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")
from common import create_connection
from common.get_ip import get_host_ip
from designer.designer_info import Designer
from common.common_log import Logger, get_log_name

current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)
this_ip = get_host_ip()

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)


class DesignerPool(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 设计师流量池，开发环境的数据是自己造的假数据
    '''

    def __init__(self):
        self.designer = Designer()

    def get_designer_pool(self):
        '''
        Author: xiaoyichao
        param {*}
        Description: 设计师所在的流量池，两个字典
        '''
        redis_connection_storage_pool = create_connection.RedisCli.connectionPool(root_config["FileLocation"]["rds_designer"],
                                                                                  section='flow_pool')
        redis_cursor = redis.StrictRedis(
            connection_pool=redis_connection_storage_pool)
        key = "designer_flow_pool_designer"
        self.designer_pool_dict = {}
        self.pool_designer_dict = {}
        designers_uid_without_son = set(self.designer.get_designer_list())
        uid_pool_redis_dict = redis_cursor.hgetall(key)
        # 构造self.designer_pool_dict
        for uid, pool in uid_pool_redis_dict.items():
            uid = int(uid.decode())
            pool = str(pool.decode())
            # log.logger.debug("uid:%s, pool:%s" % (uid, pool))
            if uid in designers_uid_without_son:
                self.designer_pool_dict[uid] = pool

        # 开发环境上需要造假数据
        if this_ip == "172.16.10.10":
            log.logger.info("开发环境上需要造流量池的假数据")
            designers_uid_list = self.designer.get_designer_list()
            self.designer_pool_dict = {}
            pool_list = ["a", "b", "c", "d"]
            for uid in designers_uid_list:
                self.designer_pool_dict[uid] = pool_list[random.randint(0, len(pool_list)-1)]
        # 构造self.pool_designer_dict
        for uid, pool in self.designer_pool_dict.items():
            if pool not in set(self.pool_designer_dict.keys()):
                self.pool_designer_dict[pool] = [uid]
            else:
                uids = self.pool_designer_dict[pool]
                uids.append(uid)
                self.pool_designer_dict[pool] = uids
        self.pool_set_sorted = sorted(set(self.pool_designer_dict.keys()))
        log.logger.info('self.pool_set_sorted: %s' %
                        str(self.pool_set_sorted))
        sum = 0
        for pool_name in self.pool_set_sorted:
            pool_len = len(self.pool_designer_dict[pool_name])
            log.logger.info("pool_name:%s" % str(pool_name) + "流量池子中设计师数量:%s" % str(pool_len))
            sum += pool_len
        log.logger.info("流量池总数据总量:%s" % sum)
        redis_cursor.close()
        redis_connection_storage_pool.disconnect()

        return self.designer_pool_dict, self.pool_designer_dict, self.pool_set_sorted


if __name__ == "__main__":
    designer_pool = DesignerPool()
    designer_pool_dict, pool_designer_dict, pool_set_sorted = designer_pool.get_designer_pool()
    print("designer_pool_dict", designer_pool_dict)
    while 1:
        try:
            designer_uid = int(input("请输入设计师的uid:"))
            print(designer_pool_dict[designer_uid])
        except Exception:
            print("设计师的uid不存在")
    # print(designer_pool_dict, pool_designer_dict, pool_set_sorted)
