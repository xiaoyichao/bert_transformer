# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  comprehensive_rank_ranking参数收集类
'''
import numpy as np

class ranking_all():
    # 搜索词
    query = ""

    # 参数收集对象
    params_collect_cls = None

    # faiss map ids numpy
    esRecallFaissMapIdsFilter = np.array([])

    # hash es数据
    hashEsDataInfos = {}

    # 请求参数全局变量
    init_request_global_var = None

    # faiss map ids list
    esRecallFaissMapIdsList = []

    # 流量池 权重
    queryVec = np.array([])

    # es 召回的内容id
    esRecallObjIdsList = []
