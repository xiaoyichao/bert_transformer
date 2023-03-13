# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/9/13
@Desciption :  搜索 热门排序 返回
'''
class run_task_for_hot_return():
    # 全量note es 召回
    allNoteInfos = {
            "total": 0,
            "rows": [],
            "max_score": 0,
        }

    # 全量 total article es 召回
    allTotalArticles = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 设计师note es 召回
    designerNotes = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 设计师 total article es 召回
    designerTotalArticles = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 绑定 wiki note es 召回
    bindWikiNotes = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 绑定 wiki total article es 召回
    bindWikiTotalArticles = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # note 向量召回
    vectNoteInfosByIds = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # total article 向量召回
    vectTotalArticleInfosByIds = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 地区 设计师 note es 召回
    areaDesignerNotes = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # 地区 设计师 total article es 召回
    areaDesignerTotalArticles = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }

    # wiki 双高池 es 召回
    wiki2HighNotes = {
        "total": 0,
        "rows": [],
        "max_score": 0,
    }


