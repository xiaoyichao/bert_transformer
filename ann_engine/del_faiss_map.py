# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-04-27 15:12:08
LastEditTime: 2021-05-10 17:07:53
Description:  实现首次全量数据的写入faiss index
'''
import os
import sys
import numpy as np
os.chdir(sys.path[0])
sys.path.append("../")
from common.common_log import Logger, get_log_name
from ann_engine.get_faiss_id import FaissId

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(all_log_name)
faissid = FaissId()


def main():
    faissid.del_faiss_map()


if __name__ == "__main__":
    main()
