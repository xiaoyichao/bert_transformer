# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  comprehensive_rank_ranking参数收集类
'''
import numpy as np

class comprehensive_rank_ranking():
    # 搜索词 切词结果
    query_splited_list = []

    # 内容 相似度
    similars_with_information_np_TopK = np.array([])

    # jaccard 文本 相似度
    jaccard4content_np = np.array([])

    # jaccard 标签 相似度
    jaccard4admin_tag_np = np.array([])

    # 单位时间交互权重
    interact_time_np = np.array([])

    # 分数权重
    admin_score_np = np.array([])

    # 流量池 权重
    pool_score_np = np.array([])

    # 用户类型
    user_types_np = np.array([])

    # 绑定wiki的内容
    is_bind_wiki_np = np.array([])

    # 内容是否绑定wiki 权重
    wiki_score_np = np.array([])

    # 意图列表
    intent_list = []

    # 完全匹配数量 0 or 1
    full_match_np =  np.array([])

    # 切词 匹配 比例
    split_match_np =  np.array([])

    # 标题长度 权重
    title_lens_weight_np = np.array([])

    # 描述长度 权重
    remark_lens_weight_np = np.array([])

    # 时间 权重
    time_weight_np = np.array([])

    # 收藏量
    favorites_nums_np = np.array([])

    # 收藏量
    favorite_weight_np = np.array([])

    # wiki 标题 相似度
    jaccard4content_wiki_name_np = np.array([])

    # ctr 得分
    ctr_scores_np = np.array([])
