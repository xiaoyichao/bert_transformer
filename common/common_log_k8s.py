# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-04-27 17:12:30
LastEditTime: 2022-03-24 17:57:23
Description:  本文件中的内容是写日志的类
'''
import os
import sys
import time
import datetime
import configparser
import logging
import socket
from logging import handlers
os.chdir(sys.path[0])
sys.path.append("../")


'''config.ini中的配置'''
current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

log_dir = root_config["log"]['log_dir']
backCount = root_config["log"]['backCount']
when = root_config["log"]['when']
hostname = socket.gethostname()
hostname = str(hostname).replace(".", "-")
today = str(datetime.date.today())

'''resource_config中的配置'''
res_config_common_file = "/home/resource_config/search_opt/common.ini"
res_config = configparser.ConfigParser()
res_config.read(res_config_common_file)
print_to_terminal = res_config["log"]["print_to_terminal"]
print_to_terminal = False if res_config["log"]["print_to_terminal"] == "False" else True
level = res_config["log"]["level"]


def get_log_name(file_abs_path, suffix: str):
    '''
    @Author: xiaoyichao
    @param {suffix:文件名后缀，比如，"all"表示所有的log的文件名。"error"表示error的文件名}
    @Description: 获取当前文件的log_name
    '''
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    relatively_path = str(file_abs_path).replace(
        str(root_path), "").strip("/").replace("/", "-").replace(".py", "")+"_"+suffix+"_"+today+"_"+str(hostname)+".log"
    return relatively_path


class Logger(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 用于写日志并打印到屏幕的类
    '''
    level_relations = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, log_name, log_dir = log_dir, when=when, backCount=int(backCount), fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        '''
        @Author: xiaoyichao
        @param {参数的使用下边有介绍}
        @Description: 初始化
        '''
        log_path = os.path.join(log_dir, log_name)
        self.logger = logging.getLogger(log_path)
        if not self.logger.handlers:  # logging 是单实例，如果不判断，多次调用的时候会导致日志数据重复
            format_str = logging.Formatter(fmt)  # 设置日志格式
            self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
            self.sh = logging.StreamHandler()  # 往屏幕上输出
            self.sh.setFormatter(format_str)  # 设置屏幕上显示的格式
            self.th = handlers.TimedRotatingFileHandler(
                filename=log_path, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            # 实例化TimedRotatingFileHandler
            # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
            self.th.setFormatter(format_str)  # 设置文件里写入的格式
            if print_to_terminal:
                self.logger.addHandler(self.sh)  # 把对象加到logger里
            self.logger.addHandler(self.th)



'''在本文件中的测试'''
if __name__ == '__main__':
    abs_path = os.path.realpath(__file__)
    all_log_name = get_log_name(abs_path, "all")
    err_log_name = get_log_name(abs_path, "error")
    log = Logger(log_name=all_log_name)
    if __name__ == '__main__':
        for i in range(1):
            time.sleep(1)
            log.logger.debug('debug:%s, %s'%(100,100))
            log.logger.info('info')
            log.logger.warning('警告')
            log.logger.error('报错')
            log.logger.critical('严重')
            Logger(log_name=err_log_name).logger.error('error')


'''其他文件中使用的方法'''
# import os
# import sys
# import time
# os.chdir(sys.path[0])

# sys.path.append("../")
# from common.common_log import Logger, get_log_name


# abs_path = os.path.realpath(__file__)
# all_log_name = get_log_name(abs_path, "all")
# err_log_name = get_log_name(abs_path, "error")
# print(err_log_name)

# log = Logger(log_name=all_log_name)


# def test3_3():
#     for i in range(5):
#         time.sleep(1)

#         log.logger.debug('debug')
#         log.logger.info('info')
#         log.logger.warning('警告')
#         log.logger.error('报错')
#         log.logger.critical('严重')
#         Logger(log_name=err_log_name).logger.error('error')


# test3_3()
