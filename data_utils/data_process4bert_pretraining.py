# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/4/26 11:56
@Desciption : 数据预处理 清洗整屋，文章，指南，经验文章的正文数据，并将其按512个字节进行切割
'''
from collections import defaultdict
import imp
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")
from common import create_connection as create_connection
import json
import re
import configparser
from common.tool import Tool
from common.split_any import split_list
from db.hhzAdmin.HhzAdmUserType import HhzAdmUserType as dbHhzAdmUserType

from common.common_log import Logger, get_log_name
from service.Init.InitResource import InitResource
from presto_con.topresto import python_presto
from common.send_message import send_msg

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

SYSTEM_IMG = 2  # 系统代替发图
TYPE_LIVE = 4  # 直播卡片
TYPE_TOTEM = 5  # 徽章

filterTypeNew = [
        SYSTEM_IMG,
        TYPE_LIVE,
        TYPE_TOTEM,
    ]

# 工具类
ToolCLs = Tool()

# limit_num = 100
limit_num = 0  # limit_num = 0 为全量数据

def SplitData(text, textMergeList):
    """
    对长文本进行切割成列表数据，先用 \n切割
    再用.分隔，最后将结果存入textMergeList中
    Args:
        text:  文本数据
        textMergeList: 用于存放切割后的列表
    Author: xuzhongjie
    Returns:

    """
    if len(text) < 3:
        return

    textOneSplit = text.split("\n")

    for textItemOne in textOneSplit:
        textTwoSplit = textItemOne.split(".")

        for textItemTwo in textTwoSplit:
            textItemTwo = removeDataInBehind(textItemTwo)
            textItemTwo = textItemTwo.strip()

            if len(textItemTwo) > 0:
                textMergeList.append(textItemTwo)


def removeDataInBehind(text):
    """
    替换text空格中的空格 和 特殊符号 \ufeff
    Args:
        text: 文本

    Returns: string
    Author: xuzhongjie
    """
    text1 = re.sub('\s+', ' ', text)
    text1 = re.sub('(\ufeff)+', ' ', text1)
    return text1

def getProcessNoteIds(limit = None):
    init_resource_cls = InitResource()
    init_resource_cls.initResource()

    # 获取B级用户
    blevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)

    # 获取note信息
    noteIds = []

    noteSqlStr = "id,uid"
    noteSqlArr = noteSqlStr.split(",")

    filterTypeStrList = [str(filterTypeItem) for filterTypeItem in filterTypeNew]

    '''使用的Hive,获取note的数据'''
    presto = python_presto()
    host = root_config["prosto"]['host']
    port = int(root_config["prosto"]['port'])
    user = root_config["prosto"]['user']
    password = root_config["prosto"]['password']
    presto_con = presto.connect(host=host,
                                port=port,
                                user=user,
                                password=password
                                )

    presto_cursor = presto_con.cursor()

    noteSql = "select " + noteSqlStr + " from oss.hhz.dwd_hhz_photos" + " where status = 1  and admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(
        filterTypeStrList) + ")"

    if limit is not None:
        noteSql += " limit " + str(limit)

    presto_cursor.execute(noteSql)
    noteInfosDb = presto_cursor.fetchall()

    # 将数字索引转换为关键词索引
    for noteInfosSampleItem in noteInfosDb:
        noteHahsInfo = {}
        for (index, noteSqlItem) in enumerate(noteSqlArr):
            data = noteInfosSampleItem[index]
            noteHahsInfo[noteSqlItem] = data

        if noteHahsInfo["uid"] in blevelUids:
            continue

        noteIds.append(noteHahsInfo["id"])

    presto_cursor.close()
    presto_con.close()

    return noteIds


def getProcessNoteInfo(ids=None, limit_num=None, use_presto=False, init_resource_cls_exterior = None):
    """
    获取处理的note信息
    Returns: {“004xz4u000076rit_0” : "好好住", "004xz4u000076rit_1" : "好好住"}
    Author: xuzhongjie
    """
    if init_resource_cls_exterior is None:
        init_resource_cls = InitResource()
        init_resource_cls.initResource()
    else:
        init_resource_cls = init_resource_cls_exterior

    # 获取B级用户
    blevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)

    # 获取note信息
    noteInfos = []

    noteSqlStr = "id,title,remark,uid"
    noteSqlArr = noteSqlStr.split(",")

    filterTypeStrList = [str(filterTypeItem) for filterTypeItem in filterTypeNew]
    if use_presto:
        '''使用的Hive,获取note的数据'''
        presto = python_presto()
        host = root_config["prosto"]['host']
        port = int(root_config["prosto"]['port'])
        user = root_config["prosto"]['user']
        password = root_config["prosto"]['password']
        presto_con = presto.connect(host=host,
                                    port=port,
                                    user=user,
                                    password=password
                                    )

        presto_cursor = presto_con.cursor()

        if limit_num and limit_num > 0:
            if ids is not None and len(ids) > 0:
                noteSql = "select " + noteSqlStr + " from  oss.hhz.dwd_hhz_photos" + " where status = 1  and " \
                        "admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(filterTypeStrList) + ") and id in ('" + "','".join(ids) + "') limit %s" % limit_num
            else:
                noteSql = "select " + noteSqlStr + " from oss.hhz.dwd_hhz_photos" + " where status = 1  and admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(
                    filterTypeStrList) + ") limit %s" % limit_num
        else:
            if ids is not None and len(ids) > 0:
                noteSql = "select " + noteSqlStr + " from oss.hhz.dwd_hhz_photos" + " where status = 1  and " \
                        "admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(filterTypeStrList) + ") and id in ('" + "','".join(ids) + "')"
            else:
                noteSql = "select " + noteSqlStr + " from oss.hhz.dwd_hhz_photos" + " where status = 1  and admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(
                    filterTypeStrList) + ") "

        presto_cursor.execute(noteSql)
        noteInfosDb = presto_cursor.fetchall()

        log.logger.info("note数量:%s" % len(noteInfosDb))

        # 将数字索引转换为关键词索引
        for noteInfosSampleItem in noteInfosDb:
            noteHahsInfo = {}
            for (index, noteSqlItem) in enumerate(noteSqlArr):
                data = noteInfosSampleItem[index]
                if noteSqlItem == "title" or noteSqlItem == "remark":
                    data = noteInfosSampleItem[index].strip()
                noteHahsInfo[noteSqlItem] = data

            if noteHahsInfo["uid"] in blevelUids:
                continue

            noteInfos.append(noteHahsInfo)

        presto_cursor.close()
        presto_con.close()
        log.logger.info("使用presto读取的note")
    else:
        '''下边这段是使用的MySQL的四个表，上边是使用的Hive的数据，是四个表合并的数据'''
        contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
        for index in range(4):
            if limit_num and limit_num > 0:
                if ids is not None and len(ids) > 0:
                    noteSql = "select " + noteSqlStr + " from hhzhome_content.hhz_photos_" + str(index) + " where status = 1  and " \
                            "admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(filterTypeStrList) + ") and id in ('" + "','".join(ids) + "') limit %s" % limit_num
                else:
                    noteSql = "select " + noteSqlStr + " from hhzhome_content.hhz_photos_" + str(index) + " where status = 1  and admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(
                        filterTypeStrList) + ") limit %s" % limit_num
            else:
                if ids is not None and len(ids) > 0:
                    noteSql = "select " + noteSqlStr + " from hhzhome_content.hhz_photos_" + str(index) + " where status = 1  and " \
                            "admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(filterTypeStrList) + ") and id in ('" + "','".join(ids) + "')"
                else:
                    noteSql = "select " + noteSqlStr + " from hhzhome_content.hhz_photos_" + str(index) + " where status = 1  and admin_score >= 12 and (length(title) > 0  or length(remark) > 0) and type not in (" + ",".join(
                        filterTypeStrList) + ") "

            content_cursor = contentDb.cursor()
            content_cursor.execute(noteSql)
            noteInfosDb = content_cursor.fetchall()
            content_cursor.close()

            log.logger.info("note数量:%s" % len(noteInfosDb))

            # 将数字索引转换为关键词索引
            for noteInfosSampleItem in noteInfosDb:
                noteHahsInfo = {}
                
                for (index, noteSqlItem) in enumerate(noteSqlArr):
                    data = noteInfosSampleItem[index]
                    if noteSqlItem == "title" or noteSqlItem == "remark":
                        data = noteInfosSampleItem[index].strip()
                    noteHahsInfo[noteSqlItem] = data

                if noteHahsInfo["uid"] in blevelUids:
                    continue

                noteInfos.append(noteHahsInfo)
        contentDb.close()
        log.logger.info("使用mysql读取的note")

    log.logger.info("note总数量:%s" % len(noteInfos))
    # log.logger.info("随便看条数据", noteInfos[0])

    # 对note文本进行过滤文本处理
    return_data4xgb = {}

    for noteInfo in noteInfos:
        return_data4xgb[noteInfo["id"]] = {"title": "", "remark": []}

        if len(noteInfo["title"]) > 0 and noteInfo["title"] is not None:
            processTitle = ToolCLs.T2S(noteInfo["title"])
            processTitle = processTitle.strip()

            return_data4xgb[noteInfo["id"]]["title"] = processTitle

        if len(noteInfo["remark"]) > 0 and noteInfo["remark"] is not None:
            # print("id", noteInfo["id"])
            # print("origin remark", noteInfo["remark"])
            # processRemark = ToolCLs.filter_tag(ToolCLs.T2S(ToolCLs.processNoteRemarkData(noteInfo["remark"])))
            processRemark = ToolCLs.filter_tag(ToolCLs.T2S(noteInfo["remark"]))
            processRemark = processRemark.strip()

            splitDataTmp = []

            SplitData(processRemark, splitDataTmp)

            return_data4xgb[noteInfo["id"]]["remark"] = splitDataTmp

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return return_data4xgb

def getProcessArticleIds():
    init_resource_cls = InitResource()
    init_resource_cls.initResource()

    # 获取B级用户
    blevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)

    # 获取article信息
    articleSqlStr = "aid,uid"

    articleSql = "select " + articleSqlStr + " from hhzhome_content.hhz_article where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(description) > 0) "

    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(articleSql)
    articleInfosDb = content_cursor.fetchall()
    content_cursor.close()

    # 整屋 数字索引 改成 关键词索引
    articleIds = []
    articleSqlArr = articleSqlStr.split(",")

    for articleInfoItem in articleInfosDb:
        articleHashInfo = {}
        for (index, articleSqlItem) in enumerate(articleSqlArr):
            articleHashInfo[articleSqlItem] = articleInfoItem[index]

        if articleHashInfo["uid"] in blevelUids:
            continue

        articleIds.append(articleHashInfo["aid"])
    return articleIds

def getProcessArticleInfo(ids=None, limit_num=None, init_resource_cls_exterior = None):
    """
    获取处理的整屋信息
    Returns: {“004xz4u000076rit_0” : "好好住", "004xz4u000076rit_1" : "好好住"}
    Author: xuzhongjie
    """
    if init_resource_cls_exterior is None:
        init_resource_cls = InitResource()
        init_resource_cls.initResource()
    else:
        init_resource_cls = init_resource_cls_exterior

    # 获取B级用户
    blevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)

    # 获取article信息
    articleSqlStr = "aid,title,description,uid"

    if limit_num and limit_num > 0:
        if ids is not None and len(ids) > 0:
            articleSql = "select " + articleSqlStr + " from hhzhome_content.hhz_article where status in (1, 5, 11)  and" \
                        " admin_score >= 60 and   (length(title) > 0  and length(description) > 0) and aid in ('" + "','".join(ids) + "') limit %s" % limit_num
        else:
            articleSql = "select " + articleSqlStr + " from hhzhome_content.hhz_article where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(description) > 0) limit %s" % limit_num
    else:
        if ids is not None and len(ids) > 0:
            articleSql = "select " + articleSqlStr + " from hhzhome_content.hhz_article where status in (1, 5, 11)  and" \
                        " admin_score >= 60 and   (length(title) > 0  and length(description) > 0) and aid in ('" + "','".join(ids) + "') "
        else:
            articleSql = "select " + articleSqlStr + " from hhzhome_content.hhz_article where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(description) > 0) "

    # print(articleSql)
    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(articleSql)
    articleInfosDb = content_cursor.fetchall()
    content_cursor.close()

    # 整屋 数字索引 改成 关键词索引
    articleInfos = []
    articleSqlArr = articleSqlStr.split(",")
    return_data4xgb = {}

    for articleInfoItem in articleInfosDb:
        articleHashInfo = {}
        for (index, articleSqlItem) in enumerate(articleSqlArr):
            if articleSqlItem == "title" or articleSqlItem == "description":
                articleHashInfo[articleSqlItem] = articleInfoItem[index].strip()
            else:
                articleHashInfo[articleSqlItem] = articleInfoItem[index]
            

        if articleHashInfo["uid"] in blevelUids:
            continue

        articleInfos.append(articleHashInfo)

    aids = []
    articleShowPhotoInfos = []
    for articleInfoDbTmp in articleInfos:
        aids.append(articleInfoDbTmp["aid"])

    # 获取整屋关联的图片 并将数字索引 转 关键词索引
    articleShowPhotoSqlStr = "aid,show_pics"
    
    log.logger.info("整屋数量:%s" % len(aids))

    aids_list = split_list(aids,100)
    for part_aids in aids_list:
        if limit_num and limit_num > 0:
            articleShowPhotoSql = "select " + articleShowPhotoSqlStr + " from p_hhzhome_content.hhzhome_content.hhz_article_show_photo where aid in ('" + "','".join(
                part_aids) + "') limit %s" % limit_num
        else:
            articleShowPhotoSql = "select " + articleShowPhotoSqlStr + " from p_hhzhome_content.hhzhome_content.hhz_article_show_photo where aid in ('" + "','".join(
                part_aids) + "')"
        articleShowInfosDb = create_connection.presto_execute(articleShowPhotoSql)
        articleShowPhotoArr = articleShowPhotoSqlStr.split(",")
        

        for articleShowInfoDbItem in articleShowInfosDb:
            articleShowHashInfo = {}
            for (index, articleShowPhotoStr) in enumerate(articleShowPhotoArr):
                articleShowHashInfo[articleShowPhotoStr] = articleShowInfoDbItem[index]
            articleShowPhotoInfos.append(articleShowHashInfo)

    print("整屋关联图片的数量", len(articleShowPhotoInfos))

    # 整屋数据整理
    hashArticleShowPhotoRemarks = {}

    for articleShowPhotoInfo in articleShowPhotoInfos:
        allRemark = ""
        aid = articleShowPhotoInfo["aid"]
        showPicInfos = json.loads(articleShowPhotoInfo["show_pics"])
        if len(showPicInfos) > 0:
            for showPicInfo in showPicInfos:
                if "remark" in showPicInfo:
                    remark = showPicInfo["remark"]
                    if remark is not None and len(remark) > 0:
                        allRemark = allRemark + " " + remark

        #     allRemark = filter_tag(T2S(processNoteRemarkData(allRemark)))
        allRemark = allRemark.strip()

        if len(allRemark) > 0:
            if aid in hashArticleShowPhotoRemarks:
                hashArticleShowPhotoRemarks[aid].append(allRemark)
            else:
                hashArticleShowPhotoRemarks[aid] = [allRemark]

    for index, articleInfo in enumerate(articleInfos):
        descs = []

        title = ToolCLs.T2S(articleInfo["title"])
        desc = ToolCLs.filter_tag(ToolCLs.T2S(articleInfo["description"]))

        if len(desc) > 0:
            splitDataTmp = []
            SplitData(desc, splitDataTmp)
            descs.extend(splitDataTmp)

        if articleInfo["aid"] in hashArticleShowPhotoRemarks:
            showPhotoRemarks = hashArticleShowPhotoRemarks[articleInfo["aid"]]
            showPhotoRemarksStr = " ".join(showPhotoRemarks)

            splitDataTmp = []
            SplitData(ToolCLs.filter_tag(ToolCLs.T2S(showPhotoRemarksStr)), splitDataTmp)
            descs.extend(splitDataTmp)

        return_data4xgb[articleInfo["aid"]] = {"title": title, "remark": descs}

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return return_data4xgb

def getProcessBlankIds():
    init_resource_cls = InitResource()
    init_resource_cls.initResource()

    # 获取B级用户
    bLevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)

    # 获取blank信息
    blankSqlStr = "bid,uid"

    blankSql = "select " + blankSqlStr + " from hhzhome_content.hhz_blank where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(content_block) > 0)"

    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(blankSql)
    blankInfosDb = content_cursor.fetchall()

    # blank 数字索引 转 关键词索引
    blankSqlArr = blankSqlStr.split(",")
    blankIds = []

    for blankInfoItem in blankInfosDb:
        blankHashInfo = {}
        for (index, blankInfoStr) in enumerate(blankSqlArr):
            blankHashInfo[blankInfoStr] = blankInfoItem[index]

        if blankHashInfo["uid"] in bLevelUids:
            continue

        blankIds.append(blankHashInfo["bid"])
    return blankIds


def getProcessBlank(ids=None, limit_num=None, init_resource_cls_exterior = None):
    """
    获取处理的经验文章信息
    Returns: {“004xz4u000076rit_0” : "好好住的文章1—1", "004xz4u000076rit_1" : "好好住的文章1-2"} ,["好好住的文章1—1", "好好住的文章1—2"], [“004xz4u000076rit_0”, 004xz4u000076rit_1]
    Author: xuzhongjie
    """
    if init_resource_cls_exterior is None:
        init_resource_cls = InitResource()
        init_resource_cls.initResource()
    else:
        init_resource_cls = init_resource_cls_exterior

    # 获取B级用户
    bLevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)
    # 获取blank信息
    blankSqlStr = "bid,title,content_block,uid"
    if limit_num and limit_num > 0:
        if ids is not None and len(ids) > 0:
            blankSql = "select " + blankSqlStr + " from hhzhome_content.hhz_blank where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(content_block) > 0) and bid in ('" + "','".join(ids) + "') limit %s" % limit_num
        else:
            blankSql = "select " + blankSqlStr + " from hhzhome_content.hhz_blank where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(content_block) > 0) limit %s" % limit_num
    else:
        if ids is not None and len(ids) > 0:
            blankSql = "select " + blankSqlStr + " from hhzhome_content.hhz_blank where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(content_block) > 0) and bid in ('" + "','".join(ids) + "') "
        else:
            blankSql = "select " + blankSqlStr + " from hhzhome_content.hhz_blank where status in (1, 5, 11)  and admin_score >= 60 and   (length(title) > 0  and length(content_block) > 0) "
    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(blankSql)
    blankInfosDb = content_cursor.fetchall()

    # blank 数字索引 转 关键词索引
    blankSqlArr = blankSqlStr.split(",")
    blankInfos = []

    for blankInfoItem in blankInfosDb:
        blankHashInfo = {}
        for (index, blankInfoStr) in enumerate(blankSqlArr):
            blankHashInfo[blankInfoStr] = blankInfoItem[index]

        if blankHashInfo["uid"] in bLevelUids:
            continue

        blankInfos.append(blankHashInfo)

    print("文章数量", len(blankInfos))

    return_data4xgb = {}
    for index, blankInfo in enumerate(blankInfos):
        allText = ""

        title = ToolCLs.T2S(blankInfo["title"])

        content_block = blankInfo["content_block"]

        if len(content_block) > 0 and content_block is not None:
            try:
                contentBlockList = json.loads(content_block)
            except Exception:
                print("经验文章数据解析失败，", blankInfo["bid"])
                continue

            for contentBlock in contentBlockList:
                if "type" in contentBlock and (contentBlock["type"] == 1 or contentBlock["type"] == 2):
                    if "content" in contentBlock:
                        content = contentBlock["content"]
                        if "text" in content:
                            text = content["text"]
                            if len(text) > 0:
                                allText += " . " + text

        desc = []
        if len(allText) > 0:
            descTmp = ToolCLs.filter_tag(ToolCLs.T2S(allText))

            SplitData(descTmp, desc)
        
        return_data4xgb[blankInfo["bid"]] = {"title": title, "remark": desc}

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return return_data4xgb

def getProcessGuideIds():
    guideSqlStr = "id,title,content"

    guideStr = "select " + guideSqlStr + " from hhzhome_content.hhz_publish_guide where status = 1"

    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(guideStr)
    guideInfosDb = content_cursor.fetchall()

    guideSqlArr = guideSqlStr.split(",")
    guideIds = []

    for guideInfoItem in guideInfosDb:
        guideHashInfo = {}
        for (index, guideInfoStr) in enumerate(guideSqlArr):
            guideHashInfo[guideInfoStr] = guideInfoItem[index]
        guideIds.append(guideHashInfo["id"])

    return guideIds

def getProcessGuide(ids=None, limit_num=None):
    """
    获取处理的指南信息
    Returns: {“004xz4u000076rit_0” : "好好住", "004xz4u000076rit_1" : "好好住"}
    Author: xuzhongjie
    """
    guideSqlStr = "id,title,content"
    if limit_num and limit_num > 0:
        if ids is not None and len(ids) > 0:
            guideStr = "select " + guideSqlStr + " from hhzhome_content.hhz_publish_guide where status = 1 and id in ('" + "','".join(ids) + "')"
        else:
            guideStr = "select " + guideSqlStr + " from hhzhome_content.hhz_publish_guide where status = 1"
    else:
        if ids is not None and len(ids) > 0:
            guideStr = "select " + guideSqlStr + " from hhzhome_content.hhz_publish_guide where status = 1 and id in ('" + "','".join(ids) + "')"
        else:
            guideStr = "select " + guideSqlStr + " from hhzhome_content.hhz_publish_guide where status = 1"

    contentDb = create_connection.MySQLCli.connect("/home/resource_config/db.ini", "db_content")
    content_cursor = contentDb.cursor()
    content_cursor.execute(guideStr)
    guideInfosDb = content_cursor.fetchall()

    guideSqlArr = guideSqlStr.split(",")
    guideInfos = []

    for guideInfoItem in guideInfosDb:
        guideHashInfo = {}
        for (index, guideInfoStr) in enumerate(guideSqlArr):
            guideHashInfo[guideInfoStr] = guideInfoItem[index]
        guideInfos.append(guideHashInfo)

    print("指南数", len(guideInfos))

    return_data4xgb = {}
    for guideInfo in guideInfos:
        content = guideInfo["content"]
        content = content.replace("▲", "")

        desc = ToolCLs.filter_tag(ToolCLs.T2S(content))

        title = ToolCLs.T2S(guideInfo["title"])

        splitDataTmp = []
        SplitData(desc, splitDataTmp)

        return_data4xgb[guideInfo["id"]] = {"title": title, "remark": splitDataTmp}

    return return_data4xgb


def get_corpus(obj_ids_list):
    from data_utils.Thread.dataProcessFromDbInPretrain import dataProcessFromDbInPretrain
    from data_utils.Thread.dataProcessInsert import dataProcessInsert
    from multiprocessing import Queue
    import multiprocessing
    '''
    Author: xiaoyichao
    param {*} self
    param {*} obj_ids_list
    Description: 获取obj_id 和 文本 组成的字典
    '''

    def get_data_by_thread(obj_ids, content_type, corpus):
        objNum = len(obj_ids)
        globalTaskNum = multiprocessing.Value('d', objNum)

        data_queue = Queue(100000)  # 存放解析数据的queue

        objIdQueue = Queue(100000)

        objType = ""
        if content_type == 1:
            objType = "blank"
        elif content_type == 2:
            objType = "note"
        elif content_type == 3:
            objType = "article"
        elif content_type == 4:
            objType = "guide"

        threadNum = 5
        thread_insert_id = 'insert_queue_1'
        threadInsert = dataProcessInsert(thread_insert_id, objIdQueue, content_type, obj_ids, threadNum)  # 启动线程
        threadInsert.start()  # 启动线程

        # 初始化请求线程
        get_info_from_db_threads = []
        get_info_name_list = ['get_info_from_db_' + str(i) for i in range(threadNum)]

        for thread_id in get_info_name_list:
            thread = dataProcessFromDbInPretrain(thread_id, objIdQueue, data_queue, objNum, content_type, objType,
                                       globalTaskNum)  # 启动线程
            thread.start()  # 启动线程
            get_info_from_db_threads.append(thread)

        NoneNum = 0
        while True:
            objid_title_remark_dict = data_queue.get(True)
            if objid_title_remark_dict is None:
                NoneNum = NoneNum + 1
                if NoneNum == threadNum:
                    break
                continue
            corpus.update(objid_title_remark_dict)

    corpus = {}
    # self.add_obj_pass_ids = []
    if len(obj_ids_list) > 0:
        # objid_content_type_dict = self.get_objid_content_type_dict(obj_ids_list)
        content_type_ids_dict = Tool.get_content_type_ids_dict(obj_ids_list)
        # {content_type:1是文章，2是note，3是整屋，4是指南}
        for content_type, obj_ids in content_type_ids_dict.items():
            if len(obj_ids) > 0:
                get_data_by_thread(obj_ids, content_type, corpus)

    return corpus

if __name__ == '__main__':
    pretraining_data_dir = "/data/search_opt_model/topk_opt/pretraining_data"

    def test(text_i, objIds, batchSize):
        batchNums = (len(objIds) // batchSize) + 1

        for batchNum in range(batchNums):
            text_i += 1

            with open(pretraining_data_dir + '/pre_data_' + str(text_i) + ".txt", 'w', encoding='utf-8') as f:
                start = batchNum * batchSize
                end = (batchNum + 1) * batchSize

                batchIds = objIds[start:end]

                hashInfos = get_corpus(batchIds)

                if len(hashInfos) > 0:
                    dataNum = len(hashInfos.keys())

                    for index, info in enumerate(hashInfos.values()):
                        isNeedAddLineBreak = False

                        title = info["title"]

                        if len(title) > 0:
                            isNeedAddLineBreak = True
                            f.write(title + "\n")

                        remarkList = info["remark"]

                        if len(remarkList) > 0:
                            for remark in remarkList:
                                if len(remark) > 0:
                                    isNeedAddLineBreak = True
                                    f.write(remark + "\n")

                        if isNeedAddLineBreak and ((dataNum - 1) != index):
                            f.write("\n")

        return text_i

    text_i = 0

    noteIds = getProcessNoteIds()
    text_i = test(text_i, noteIds, 100000)

    articleIds = getProcessArticleIds()
    text_i = test(text_i, articleIds, 10000)

    blankIds = getProcessBlankIds()
    text_i = test(text_i, blankIds, 10000)

    guideIds = getProcessGuideIds()
    test(text_i, guideIds, 10000)

