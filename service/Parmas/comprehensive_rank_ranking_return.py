# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  comprehensive_rank_ranking 返回值
'''
import numpy as np

class comprehensive_rank_ranking_return():
    # 排序得分
    rank_scores_np = np.array([])

    # 排序得分
    rank_scores_with_wiki_pool_np = np.array([])

    # 内容 相似度 得分
    similars_with_information_np_format = np.array([])

    # jaccard admin_tag 得分
    jaccard4admin_tag_np_format = np.array([])

    # 交互时间比 得分
    interact_time_np_format = np.array([])

    # 内容质量 得分
    admin_score_np_format = np.array([])

    # 流量池 得分
    pool_score_np_format = np.array([])

    # 内容绑定 wiki 得分
    wiki_score_np_format = np.array([])

    # 内容与搜索词 完全匹配 得分
    jaccard4content_full_match_np_format = np.array([])

    # 内容与 搜索词分词后 匹配 得分
    jaccard4content_split_match_np_format = np.array([])

    # 标题长度得分
    title_lens_np_format = np.array([])

    # 描述长度得分
    remark_lens_np_format = np.array([])

    # 收藏量得分
    favorite_np_format = np.array([])

    # ctr 得分
    ctr_score_np_format = np.array([])
