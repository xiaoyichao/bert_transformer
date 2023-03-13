# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  match 返回
'''
import numpy as np

class matching_return():
    # jaccard content 相似度
    jaccard4content_np = np.array([])

    # jaccard admin_tag 相似度
    jaccard4admin_tag_np = np.array([])

    # 交互时间比 得分
    interact_time_np = np.array([])

    # 质量 得分
    admin_score_np = np.array([])

    # 设计师流量 得分
    pool_score_np = np.array([])

    # 绑定商品 得分
    wiki_score_np = np.array([])

    # 用户类型
    user_types_np = np.array([])

    # 内容对应 用户类型
    hash_obj_user_type = {}

    # 内容对应 jaccard_sim
    hash_obj_id_jaccard_sim = {}

    # 内容对应 是否 关联 wiki
    hash_obj_related_wiki = {}

    # 完全匹配 得分
    full_match_np = np.array([])

    # 切词匹配 得分
    split_match_np = np.array([])

    # 标题长度 得分
    title_lens_weight_np = np.array([])

    # 描述长度 得分
    remark_lens_weight_np = np.array([])

    # 时间 得分
    time_weight_np = np.array([])

    # 收藏数量
    favorites_nums_np = np.array([])

    # 设计师流量等级
    pool_data_list = []

    # 内容绑定 wiki 得分
    jaccard4content_wiki_name_np = np.array([])

    # 内容对应 切词匹配得分
    hash_obj_id_jaccard_split_match_sim = {}

    # 内容是否绑定了wiki
    is_bind_wiki_np = np.array([])

    # 收藏权重
    favorite_weight_np = np.array([])

    # 内容对应 设计师流量等级
    hash_obj_designer_flow = {}

    # 设计师 主服务地区
    hash_main_service_area = {}
