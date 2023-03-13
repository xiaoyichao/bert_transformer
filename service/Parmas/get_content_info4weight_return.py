# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  搜索 热门排序 返回
'''
class get_content_info4weight_return():
    # wiki 得分
    wiki_score_list = []

    # 设计师流量等级
    pool_data_list = []

    # 点赞数
    like_nums = []

    # 收藏数
    favorite_nums = []

    # 评论数
    comment_nums = []

    # 发布时间
    publish_time = []

    # 质量得分
    admin_scores = []

    # 内容分词
    contents_splited_list = []

    # 标签
    content_admin_tags = []

    # 文本长度
    contents_lens = []

    # 文本列表
    contents_str_list = []

    # 用户类型
    user_types = []

    # 内容对应用户类型
    hash_obj_user_type = {}

    # 内容对应 是否绑定wiki
    hash_obj_related_wiki = {}

    # 标题长度
    title_lens = []

    # 描述长度
    remark_lens = []

    # 内容绑定 wiki 的 标题切词
    rela_wiki_names_splited_list = []

    # 内容绑定 wiki 的 标题
    rela_wiki_names_str_list = []

    #内容是否 绑定 wiki
    is_bind_wiki_list = []

    # 内容对应 设计师流量等级
    hash_obj_designer_flow = {}

    # 设计师 主服务地区
    hash_main_service_area = {}

    # 对标题的 切词 结果
    hash_obj_split_title_for_ai = {}

    # 对描述的 切词 结果
    hash_obj_split_remark_for_ai = {}
