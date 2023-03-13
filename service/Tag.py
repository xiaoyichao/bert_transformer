# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 22:42
@Desciption :  hhz admin tag相关操作
'''
import os
import sys
from db.hhzTag.HhzTag import HhzTag

os.chdir(sys.path[0])
sys.path.append("../")
tagsFilePath = "/home/adm_rsync_dir/tags/tags_all"

class Tag(object):
    # 标签集合
    adminTags = set()

    @classmethod
    def getAllAdminTags(cls):
        """
        获取所有标签
        Returns:

        """
        if len(cls.adminTags) == 0:
            adminTags = set([line.strip() for line in open(
                tagsFilePath, 'r', encoding='utf-8').readlines()])
            cls.adminTags = adminTags
        return cls.adminTags

    @classmethod
    def handleIsAdminTagResult(cls, splitWords):
        """
        将切词结果分成两部分，分别是 标签词集合 和 非标签词集合，并且返回
        Args:
            splitWords: 词汇的列表

        Returns:

        """
        splitWordsAdminTag = set()
        splitWordsNotAdminTag = set()

        adminTags = cls.getAllAdminTags()

        # 这种结构 时间复杂度 是O(1)
        hashAdminTags = {}
        for adminTag in adminTags:
            hashAdminTags[adminTag] = True

        for splitWord in splitWords:
            if splitWord in hashAdminTags:
                splitWordsAdminTag.add(splitWord)
            else:
                splitWordsNotAdminTag.add(splitWord)

        return list(splitWordsAdminTag), list(splitWordsNotAdminTag)

    @classmethod
    def get_all_community_tags(cls, init_resource_cls):
        """
        获取 社区 标签 和  标签别名
        Args:

        Returns: set()
        """
        all_admin_tags_infos = HhzTag.get_all_admin_tags(init_resource_cls)

        community_id = 0
        for all_admin_tags_info in all_admin_tags_infos:
            if all_admin_tags_info["tag"] == "社区标签":
                community_id = all_admin_tags_info["id"]

        community_tags = cls.get_tags(all_admin_tags_infos, community_id)
        return_tags = set()

        for community_tag in community_tags:
            if "tag" in community_tag:
                tag_tmp = community_tag["tag"].strip()
                if len(tag_tmp) > 0:
                    return_tags.add(tag_tmp)

            if "tag_alias" in community_tag:
                tag_alias = community_tag["tag_alias"].strip()
                tag_alias_list = tag_alias.split(",")

                for tag_alias_item in tag_alias_list:
                    tag_alias_item_tmp = tag_alias_item.strip()
                    if len(tag_alias_item_tmp) > 0:
                        return_tags.add(tag_alias_item_tmp)

        return return_tags

    @classmethod
    def get_tags(cls, tagsInfo, pid):
        childs = cls.find_son(tagsInfo, pid)

        for child in childs:
            son = cls.find_son(tagsInfo, child["id"])
            if len(son) > 0:
                childs.extend(son)

        return childs

    @classmethod
    def find_son(cls, tagsInfo, id):
        child = []
        for tagInfo in tagsInfo:
            if tagInfo["parent_id"] == id:
                child.append(tagInfo)
        return child



# if __name__ == '__main__':
#     from service.Init.InitResource import InitResource
#     InitResource.initResource()
#
#     print(Tag.get_all_community_tags())