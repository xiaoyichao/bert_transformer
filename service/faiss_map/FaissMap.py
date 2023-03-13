# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/2 14:43
@Desciption : 提供 和faiss_map表 操作的服务
'''
import os
import sys
from db.hhzSearch.FaissMap import FaissMap as dbFaissMap
from common.common_log import Logger, get_log_name
import configparser
from cache.SearchContent import SearchContent
from service.Init.InitResource import InitResource

dir_name = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(dir_name, "../../config/config.ini")

root_config = configparser.ConfigParser()
root_config.read(config_path)

os.chdir(sys.path[0])
sys.path.append("../../")


absPath = os.path.realpath(__file__)
logName = get_log_name(absPath, "all")
log = Logger(log_name=logName)


class FaissMap(object):
    # 存放 faiss_map表中id 和 内容id-passageId 的关系
    Id2ObjIdAndPassageID = {}
    # 存放 内容id 和 内容id-passageId-faiss_map表中id 的关系
    ObjId2PassageIdAndId = {}

    @classmethod
    def buildFaissMapCache(cls, init_resource_cls):
        """
        将faiss_map 表中的数据生成缓存
        Returns:

        """
        FaissMapInfos = dbFaissMap.getAllData(init_resource_cls)

        for FaissMapInfo in FaissMapInfos:
            cls.Id2ObjIdAndPassageID[int(FaissMapInfo["id"])] = str(FaissMapInfo["obj_id"]) + "-" + str(FaissMapInfo["obj_passage_id"])

            PassageIdAndId = str(FaissMapInfo["obj_passage_id"]) + "-" + str(FaissMapInfo["id"])
            if FaissMapInfo["obj_id"] in cls.ObjId2PassageIdAndId:
                cls.ObjId2PassageIdAndId[FaissMapInfo["obj_id"]].append(PassageIdAndId)
            else:
                cls.ObjId2PassageIdAndId[FaissMapInfo["obj_id"]] = [PassageIdAndId]

    @classmethod
    def getObjIdsByIds(cls, ids, needEmpty=False, needRepeatObjId=False):
        """
        通过ids获取内容id
        Args:
            ids:  faiss_map 表中的 id

        Returns: list

        """
        objIds = []
        objIdIsExist = {}

        if len(ids) > 0:
            for id in ids:
                idInt = int(id)
                ObjIdAndPassageID = ""
                # id是否存在于 Id2ObjIdAndPassageID
                if idInt in cls.Id2ObjIdAndPassageID:
                    ObjIdAndPassageID = cls.Id2ObjIdAndPassageID[idInt]

                if ObjIdAndPassageID:
                    ObjIdAndPassageIDList = ObjIdAndPassageID.split("-")
                    # 因为想要返回列表数据，所以将已存放的内容id事先存于objIdIsExist
                    if ObjIdAndPassageIDList[0] not in objIdIsExist:
                        objIds.append(ObjIdAndPassageIDList[0])
                        objIdIsExist[ObjIdAndPassageIDList[0]] = True
                    elif needRepeatObjId:
                        objIds.append(ObjIdAndPassageIDList[0])
                elif needEmpty:
                    objIds.append("")

        log.logger.debug("objIds lenth: " + str(len(objIds)))
        return objIds

    @classmethod
    def getIdByObjIds(cls, objIds):
        """
        通过内容ids获取 faiss_map表中的id
        Args:
            objIds: 内容ids

        Returns: set()

        """
        ids = set()

        if len(objIds) > 0:
            for objId in objIds:
                if objId in cls.ObjId2PassageIdAndId:
                    # 一个内容id关可能联多个 PassageId
                    PassageIdAndIdList = cls.ObjId2PassageIdAndId[objId]
                    for PassageIdAndId in PassageIdAndIdList:
                        PassageIdAndIdList = PassageIdAndId.split("-")
                        id = PassageIdAndIdList[1]

                        if len(id) > 0:
                            ids.add(id)

        return list(ids)

    @classmethod
    def get_faissmap_update_delete_ids(cls):
        """
        获取增量更新的 更新 ids 和 删除 ids

        Returns: list 更新的ids, list 删除的ids

        """
        init_resource_cls = InitResource()
        init_resource_cls.initResource()

        update_obj_ids = SearchContent.get_faissmap_update(init_resource_cls)
        delete_obj_ids = SearchContent.get_faissmap_delete(init_resource_cls)

        return update_obj_ids, delete_obj_ids
