# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-07-30 14:04:25
LastEditTime: 2021-08-10 14:30:14
Description: 
'''
from service.Weight.match_operate import MatchingOp
import numpy as np
import os
import time
import configparser
# from service.data_process import DataProcess
from service.Weight.calc_weight import CalcWeight
# from service.faiss_map.FaissMap import FaissMap
# from service.Init.InitService import InitService
from common.common_log import Logger, get_log_name
# from db.hhzMember.Member import Member
from service.faiss_map.FaissMap import FaissMap as sFaissMap
from service.faiss_map.ObjPassageVects import ObjPassageVects
from service.faiss_map.ObjVects import ObjVects

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)


class RelatedRank(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 相关性排序 的类
    '''
    TopK_num = 600  # 按照相似度排序后的结果，取TopK

    @classmethod
    def get_similars(self, params_collect_cls, init_request_global_var, esRecallFaissMapIdsList, queryVec):
        """
        获取搜索词 和 内容 的语义相似度
        Args:
            esRecallFaissMapIdsList: es 召回的faiss map ids
            query_vec: 搜索词 向量
        Returns:

        """

        # 通过 FaissMapId 获取向量
        starttime = time.time()
        contentVectors = ObjPassageVects.getVectsByFaissMapIds(esRecallFaissMapIdsList, -(queryVec[0]))
        endtime = time.time()
        use_time = endtime - starttime
        log.logger.info("通过 FaissMapId 获取向量 运行时间：{:.10f} s".format(use_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 将数据预处理成可以运算的格式
        contentVectors = contentVectors.reshape(-1, 512)
        queryVecList = queryVec.reshape(-1, 512)

        # 计算搜索词向量 和 内容向量 的点积结果
        starttime = time.time()
        log.logger.info("开始计算余弦相似度")

        # 构造假数据
        # need_shape = contentVectors.shape
        # print(need_shape)
        # contentVectors = np.random.random(need_shape)
        # contentVectors = np.random.random((3, 512))

        # need_shape = queryVecList.shape
        # print(need_shape)
        # queryVecList = np.random.random(need_shape)
        # queryVecList = np.random.random((1, 512))

        # 计算余弦相似度
        # cosine_similars = np.array(cosine_similarity(queryVecList, contentVectors)[0])
        cosine_similars = CalcWeight.cosine_similarity_hhz(queryVecList, contentVectors)[0]
        endtime = time.time()
        log.logger.info("计算余弦相似度 运行时间：{:.10f} s".format(endtime - starttime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 对同一个内容id 存在多个段的情况 计算相似度平均值
        starttime = time.time()
        cosine_similars = CalcWeight.calcMeanQuerySimilar(cosine_similars, esRecallFaissMapIdsList)
        endtime = time.time()
        log.logger.info(
            "计算 Mean cosine_similars 运行时间：{:.10f} s".format(endtime - starttime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        return cosine_similars

    @classmethod
    def get_similars_new(self, params_collect_cls, init_request_global_var, objIdsList, queryVec):
        """
        获取搜索词 和 内容 的语义相似度
        Args:
            objIdsList: 内容id 列表
            query_vec: 搜索词 向量
        Returns:

        """

        # 通过 FaissMapId 获取向量
        starttime = time.time()
        contentVectors = ObjVects.getVectsByIds(objIdsList, -(queryVec[0]))
        endtime = time.time()
        use_time = endtime - starttime
        log.logger.info("通过 objIdsList 获取向量 运行时间：{:.10f} s".format(
            use_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        # 将数据预处理成可以运算的格式
        contentVectors = contentVectors.reshape(-1, 768)
        queryVecList = queryVec.reshape(-1, 768)

        # 计算搜索词向量 和 内容向量 的点积结果
        starttime = time.time()
        log.logger.info("开始计算余弦相似度")

        # 计算余弦相似度
        cosine_similars = CalcWeight.cosine_similarity_hhz(queryVecList, contentVectors)[0]
        endtime = time.time()
        log.logger.info("计算余弦相似度 运行时间：{:.10f} s".format(
            endtime - starttime) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        return cosine_similars

    @classmethod
    def ranking_all_new(cls, ranking_all_cls):
        """
        根据相似度和信息量的乘积进行排序
        Args:
        Returns:

        """
        query = ranking_all_cls.query
        params_collect_cls = ranking_all_cls.params_collect_cls

        objIdsList = ranking_all_cls.esRecallObjIdsList

        hashEsDataInfos = ranking_all_cls.hashEsDataInfos
        init_request_global_var = ranking_all_cls.init_request_global_var
        queryVec = np.array([ranking_all_cls.queryVec])

        # 获取余弦相似度
        cosine_similars = cls.get_similars_new(params_collect_cls, init_request_global_var, objIdsList, queryVec)

        similars_with_information_np = np.array(cosine_similars)

        return similars_with_information_np, cosine_similars

    @classmethod
    def ranking_all(cls, ranking_all_cls):
        """
        根据相似度和信息量的乘积进行排序
        Args:
        Returns:

        """
        query = ranking_all_cls.query
        params_collect_cls = ranking_all_cls.params_collect_cls
        esRecallFaissMapIdsFilter = ranking_all_cls.esRecallFaissMapIdsFilter
        hashEsDataInfos = ranking_all_cls.hashEsDataInfos
        init_request_global_var = ranking_all_cls.init_request_global_var
        esRecallFaissMapIdsList = ranking_all_cls.esRecallFaissMapIdsList
        queryVec = ranking_all_cls.queryVec

        start_time = time.time()
        # 获取余弦相似度
        cosine_similars = cls.get_similars(params_collect_cls, init_request_global_var, esRecallFaissMapIdsList, queryVec)
        # 获取内容的信息量
        information_amounts = MatchingOp.get_information_amounts(params_collect_cls, esRecallFaissMapIdsFilter, hashEsDataInfos, query, init_request_global_var)

        # 计算相似度和信息量的乘积
        similars_with_information_np = cosine_similars * information_amounts
        # similars_with_information_np = 2*cosine_similars + information_amounts
        end_time = time.time()
        log.logger.info("计算 similars 和 information_amounts 运行时间：{:.10f} s".format(
            end_time - start_time) + " uid:" + params_collect_cls.uid + " query:" + params_collect_cls.query + " unique_str:" + init_request_global_var.UNIQUE_STR)

        return similars_with_information_np, cosine_similars

    @classmethod
    def get_ranked_topk(cls, cosine_similars, esRecallFaissMapIdsFilter):
        """
        获取topK
        Args:
            cosine_similars: 相似度*信息量的排序分数
            esRecallFaissMapIdsFilter: faiss map ids
        Returns:

        """
        # 获取按照相似度值排序 的索引位置
        sortIndex = np.argsort(-cosine_similars)

        TopK_num_index = np.array(sortIndex) < cls.TopK_num
        # negative_index = similars_with_information_np < 0
        # need_save_index = negative_index & TopK_num_index
        need_save_index = TopK_num_index

        # 获取排序后的FaissMapId
        sortedFaissMapIds = esRecallFaissMapIdsFilter[sortIndex]

      # 获取按照相似度值排序的前n条
        sortedFaissMapIds_TopK = sortedFaissMapIds[:cls.TopK_num]

        return sortedFaissMapIds_TopK, need_save_index

    @classmethod
    def filter_data_not_TopK(cls, need_save_index, esRecallFaissMapIdsList, similars_with_information_np):
        # print("need_save_index", need_save_index)
        # print("esRecallFaissMapIdsList", esRecallFaissMapIdsList)
        return esRecallFaissMapIdsList[need_save_index], similars_with_information_np[need_save_index]
