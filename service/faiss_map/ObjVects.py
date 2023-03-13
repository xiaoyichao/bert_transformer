# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-05-18 14:16:43
LastEditTime: 2021-06-21 15:59:34
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/8 17:03
@Desciption :  加载内容向量到内存中
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")
from common.common_log import Logger, get_log_name
from rw_vector.vecbin4recall import VecBin
import numpy as np

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)


class ObjVects(object):
    # 全局段内容向量
    IdVecDict = {}

    @classmethod
    def loadObjVects(cls):
        """
        加载段向量内容进缓存
        Returns:

        """
        if cls.IdVecDict == {}:
            readvec = VecBin()
            cls.IdVecDict = readvec.load_get_id_vec_dict()

    @classmethod
    def getVectsByIds(cls, ids, reverseQueryVector):
        """
        通过  ids 获取向量
        Args:
            ids: 内容 ids
            reverseQueryVector: 搜索词向量的反向向量

        Returns:

        """
        returnData = []

        for id in ids:
            if id in cls.IdVecDict:
                returnData.append(cls.IdVecDict[id])
            else:
                returnData.append(reverseQueryVector)

        return np.array(returnData)

if __name__ == "__main__":
    ObjVects.loadObjVects()
    print(ObjVects.IdVecDict.keys())



