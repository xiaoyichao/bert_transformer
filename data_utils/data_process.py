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
    对长文本进行切割成列表数据，每一段最长为512个字节 先用 \n切割 段长度还是超过512
    用.分隔，还是超过用,分隔，最后将结果存入textMergeList中
    Args:
        text:  文本数据
        textMergeList: 用于存放切割成512个字节的列表
    Author: xuzhongjie
    Returns:

    """
    textOneMergeData = ""
    textOneSplit = text.split("\n")

    for textItem in textOneSplit:
        if len(textItem) <= 512:
            textOneMergeData = splitChildData(textItem, textMergeList, textOneMergeData)
        else:
            if len(textOneMergeData) > 0:
                textOneMergeData = removeDataInBehind(textOneMergeData)
                textMergeList.append(textOneMergeData)
                textOneMergeData = ""

            textTwoMergeData = ""
            textTwoSplit = textItem.split(".")

            for textItem in textTwoSplit:
                if len(textItem) <= 512:
                    textTwoMergeData = splitChildData(textItem, textMergeList, textTwoMergeData)
                else:
                    if len(textTwoMergeData) > 0:
                        textTwoMergeData = removeDataInBehind(textTwoMergeData)
                        textMergeList.append(textTwoMergeData)

                    textThreeMergeData = ""
                    textThreeSplit = textItem.split(",")

                    for textItem in textThreeSplit:
                        if len(textItem) <= 512:
                            textThreeMergeData = splitChildData(textItem, textMergeList, textThreeMergeData)
                        else:
                            if len(textThreeMergeData) > 0:
                                textThreeMergeData = removeDataInBehind(textThreeMergeData)
                                textMergeList.append(textThreeMergeData)
                                textThreeMergeData = ""
                            textFourMergeData = ""
                            textFourSplit = textItem.split(",")
                            for textItem in textFourSplit:
                                if len(textItem) <= 512:
                                    textFourMergeData = splitChildData(textItem, textMergeList, textThreeMergeData)
                            if len(textFourMergeData) > 0:
                                textFourMergeData = removeDataInBehind(textFourMergeData)
                                textMergeList.append(textFourMergeData)

                    if len(textThreeMergeData) > 0:
                        textThreeMergeData = removeDataInBehind(textThreeMergeData)
                        textMergeList.append(textThreeMergeData)

            if len(textTwoMergeData) > 0:
                textTwoMergeData = removeDataInBehind(textTwoMergeData)
                textMergeList.append(textTwoMergeData)

    if len(textOneMergeData) > 0:
        textOneMergeData = removeDataInBehind(textOneMergeData)
        textMergeList.append(textOneMergeData)


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


def splitChildData(text, textMergeList, textMergeData):
    """
    文本字符串拼接，拼接后小于512个字节 放入textMergeList 大于512个字节 放入拼接前的数据
    Args:
        text: 用于拼接的文本
        textMergeList: 用于存放切割后的数据
        textMergeData: 已经拼接好的文本

    Returns: list
    Author: xuzhongjie
    """
    text = text.strip()
    textMergeDataTmp = textMergeData + " " + text
    textMergeDataTmp = textMergeDataTmp.strip()
    if len(textMergeDataTmp) <= 512:
        textMergeData = textMergeDataTmp
    else:
        textMergeData = removeDataInBehind(textMergeData)
        textMergeList.append(textMergeData)
        textMergeData = text

    return textMergeData

def getProcessNoteIds():
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


def getProcessNoteInfo(ids=None, limit_num=None, use_presto=True, init_resource_cls_exterior = None):
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
        log.logger.info("开始使用presto读取的note")
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
    processNoteInfos = []
    return_data4xgb = {}

    for noteInfo in noteInfos:
    # for i, noteInfo in enumerate(noteInfos):
        
        # if i/100 == 0:
        #     log.logger.info("note 数据解析第%s条"%str(i))
        #     send_msg("note 数据解析第%s条数据"%str(i))
        processNoteInfo = {
            "id": noteInfo["id"],
            "title": "",
            "remark": ""
        }
        return_data4xgb[noteInfo["id"]] = {"title": "", "remark": "", "uid":noteInfo["uid"]}

        if len(noteInfo["title"]) > 0 and noteInfo["title"] is not None:
            # print("id", noteInfo["id"])
            # print("origin title", noteInfo["title"])
            processTitle = ToolCLs.T2S(noteInfo["title"])
            processTitle = processTitle.strip()
            # print("title process:", processTitle)
            # print("----------\n")
            processNoteInfo["title"] = processTitle
            return_data4xgb[noteInfo["id"]]["title"] = processTitle

        if len(noteInfo["remark"]) > 0 and noteInfo["remark"] is not None:
            # print("id", noteInfo["id"])
            # print("origin remark", noteInfo["remark"])
            # processRemark = ToolCLs.filter_tag(ToolCLs.T2S(ToolCLs.processNoteRemarkData(noteInfo["remark"])))
            processRemark = ToolCLs.filter_tag(ToolCLs.T2S(noteInfo["remark"]))
            processNoteInfo["remark"] = processRemark.strip()
            processNoteInfo["remark"] = re.sub('\s+', ' ', processNoteInfo["remark"])
            return_data4xgb[noteInfo["id"]]["remark"] = processNoteInfo["remark"]
            # print("remark process:", processNoteInfo["remark"])
            # print("----------\n")

        if len(processNoteInfo["title"]) > 0 or len(processNoteInfo["remark"]) > 0:
            processNoteInfos.append(processNoteInfo)

    # log.logger.info("随便看条数据", processNoteInfos[0])

    # 按照512个字节切割后
    hashNoteProcessRemark = {}
    for processNoteInfo in processNoteInfos:
        if len(processNoteInfo["remark"]) > 0:
            splitNoteData = []
            SplitData(processNoteInfo["remark"], splitNoteData)
            hashNoteProcessRemark[processNoteInfo["id"]] = splitNoteData
            if len(processNoteInfo["title"]) > 0:
                hashNoteProcessRemark[processNoteInfo["id"]].insert(0, processNoteInfo["title"])

    returnData = {}
    return_data4rank = defaultdict(str)
    docs = []
    ids = []
    for noteId, NoteProcessRemarkList in hashNoteProcessRemark.items():
        for index, NoteProcessRemarkItem in enumerate(NoteProcessRemarkList):
            indexStr = str(noteId) + "_" + str(index)
            returnData[indexStr] = NoteProcessRemarkItem
            return_data4rank[str(noteId)] += NoteProcessRemarkItem
            docs.append(NoteProcessRemarkItem)
            ids.append(indexStr)

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return returnData, docs, ids, return_data4rank, return_data4xgb

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

    content_cursor = contentDb.cursor()
    aids_list = split_list(aids,100)
    for part_aids in aids_list:
        if limit_num and limit_num > 0:
            articleShowPhotoSql = "select " + articleShowPhotoSqlStr + " from hhzhome_content.hhz_article_show_photo where aid in ('" + "','".join(
                part_aids) + "') limit %s" % limit_num
        else:
            articleShowPhotoSql = "select " + articleShowPhotoSqlStr + " from hhzhome_content.hhz_article_show_photo where aid in ('" + "','".join(
                part_aids) + "')"

        content_cursor.execute(articleShowPhotoSql)
        articleShowPhotos = content_cursor.fetchall()

        articleShowPhotoArr = articleShowPhotoSqlStr.split(",")

        for articleShowInfoDbItem in articleShowPhotos:
            articleShowHashInfo = {}
            for (index, articleShowPhotoStr) in enumerate(articleShowPhotoArr):
                articleShowHashInfo[articleShowPhotoStr] = articleShowInfoDbItem[index]
            articleShowPhotoInfos.append(articleShowHashInfo)
    content_cursor.close()
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
        mergeData = []
        descs = []

        title = ToolCLs.T2S(articleInfo["title"])
       
        if len(title) > 0:
            mergeData.insert(0, title)

        desc = ToolCLs.filter_tag(ToolCLs.T2S(articleInfo["description"]))
        if len(desc) > 0:
            splitDataTmp = []
            SplitData(desc, splitDataTmp)
            mergeData.extend(splitDataTmp)
            descs.extend(splitDataTmp)

        if articleInfo["aid"] in hashArticleShowPhotoRemarks:
            showPhotoRemarks = hashArticleShowPhotoRemarks[articleInfo["aid"]]
            showPhotoRemarksStr = " ".join(showPhotoRemarks)
            splitDataTmp = []
            SplitData(ToolCLs.filter_tag(ToolCLs.T2S(showPhotoRemarksStr)), splitDataTmp)
            mergeData.extend(splitDataTmp)
            descs.extend(splitDataTmp)
        #         for showPhotoRemark in showPhotoRemarks:
        #             if len(showPhotoRemark) > 0 :
        #                 splitDataTmp = []
        #                 SplitData(showPhotoRemark, splitDataTmp)
        #                 mergeData.extend(splitDataTmp)

        if len(mergeData) > 0:
            articleInfos[index]["merge_data"] = mergeData
        return_data4xgb[articleInfo["aid"]] = {"title": title, "remark": ''.join(descs), "uid":articleInfo["uid"]}

    returnData = {}
    return_data4rank = defaultdict(str)
    docs = []
    ids = []
    for articleInfo in articleInfos:
        for index, articleItem in enumerate(articleInfo["merge_data"]):
            indexStr = str(articleInfo["aid"]) + "_" + str(index)
            returnData[indexStr] = articleItem
            return_data4rank[str(articleInfo["aid"])] += articleItem
            docs.append(articleItem)
            ids.append(indexStr)

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return returnData, docs, ids, return_data4rank, return_data4xgb

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

    hashBlankRemark = {}
    return_data4xgb = {}
    for index, blankInfo in enumerate(blankInfos):
        allText = ""
        desc = ""

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
                                allText += "" + text
        if len(allText) > 0:
            splitDataTmp = []
            desc = ToolCLs.filter_tag(ToolCLs.T2S(allText))

            SplitData(ToolCLs.filter_tag(ToolCLs.T2S(allText)), splitDataTmp)
            splitDataTmp.insert(0, title)
            hashBlankRemark[blankInfo["bid"]] = splitDataTmp
        
        return_data4xgb[blankInfo["bid"]] = {"title": title, "remark": desc, "uid": blankInfo["uid"]}

    returnData = {}
    return_data4rank = defaultdict(str)
    docs = []
    ids = []
    for bid, hashBlankRemarkList in hashBlankRemark.items():
        for index, hashBlankRemarkItem in enumerate(hashBlankRemarkList):
            indexStr = str(bid) + "_" + str(index)
            returnData[indexStr] = hashBlankRemarkItem
            return_data4rank[str(bid)] += hashBlankRemarkItem
            docs.append(hashBlankRemarkItem)
            ids.append(indexStr)

    if init_resource_cls_exterior is None:
        init_resource_cls.closeResource()

    return returnData, docs, ids, return_data4rank, return_data4xgb

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
    # init_resource_cls = InitResource()
    # init_resource_cls.initResource()
    # # 获取B级用户,指南没有B级用户
    # bLevelUids = dbHhzAdmUserType.getBLevelUserIds(init_resource_cls)
    # 获取guide信息
    guideSqlStr = "id,title,content,uid"
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

    hashGuideInfoRemark = {}
    return_data4xgb = {}
    for guideInfo in guideInfos:
        content = guideInfo["content"]
        content = content.replace("▲", "")
        contentList = []
        desc = ToolCLs.filter_tag(ToolCLs.T2S(content))
        SplitData(desc, contentList)

        title = ToolCLs.T2S(guideInfo["title"])
        contentList.insert(0, title)
        if len(contentList) > 0:
            hashGuideInfoRemark[guideInfo["id"]] = contentList
        return_data4xgb[guideInfo["id"]] = {"title": title, "remark": desc, "uid":guideInfo["uid"]}

    returnData = {}
    return_data4rank = defaultdict(str)
    docs = []
    ids = []
    for id, GuideInfoRemarkList in hashGuideInfoRemark.items():
        for index, GuideInfoRemarkItem in enumerate(GuideInfoRemarkList):
            indexStr = str(id) + "_" + str(index)
            returnData[indexStr] = GuideInfoRemarkItem
            return_data4rank[str(id)] += GuideInfoRemarkItem
            docs.append(GuideInfoRemarkItem)
            ids.append(indexStr)

    # init_resource_cls.closeResource()
    return returnData, docs, ids, return_data4rank, return_data4xgb


if __name__ == '__main__':
    # returnData, docs, ids, return_data4rank, return_data4xgb = getProcessGuide(["000000202000097p"])
    # print(return_data4rank)

    # InitResource.initResource()
    # processGuide = getProcessGuide(["000000202000097p"])
    # print(processGuide)

    # returnData, docs, ids, return_data4rank, return_data4xgb = getProcessBlank(["00000ob0500078xm"])
    # print(return_data4rank)

    returnData, docs, ids, return_data4rank, return_data4xgb = getProcessArticleInfo()
    print(len(return_data4rank))

    # returnData, docs, ids, return_data4rank, return_data4xgb = getProcessNoteInfo(["0000002000000980", "0011uqb00002kex5"])
    # print(return_data4rank)
    #
    # with open("id.txt", 'w') as name:
    #     for id in ids:
    #         name.write(str(id) + "\n")

    # import time
    #
    # redisConnectionStoragePool = create_connection.RedisCli.connectionPool(
    #     "/home/resource_config/search_opt/redis.ini",
    #     section='search_test')
    # import redis
    # redis_cursor = redis.StrictRedis(
    #     connection_pool=redisConnectionStoragePool)
    #
    #
    # pipe = redis_cursor.pipeline(transaction=False)
    # keys = set()
    # for line in open(
    #         "id.txt", 'r', encoding='utf-8').readlines():
    #     idData = line.strip()
    #     if idData:
    #         pipe.get(idData)
    #         keys.add(idData)
    #
    # starttime = time.time()
    # # result = pipe.execute()
    # result = redis_cursor.hmget("test", keys)
    # endtime = time.time()
    #
    # dtime = endtime - starttime
    # print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
    # print(len(result))
