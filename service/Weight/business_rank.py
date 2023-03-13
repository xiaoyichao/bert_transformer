# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-07-30 13:48:13
LastEditTime: 2021-08-10 15:18:33
Description: 
'''
from db.hhzMember.Member import Member
from service.Init.InitService import InitService
import random
import numpy as np

class BusinessRanking:
    '''
    Author: zhongjie
    param {*}
    Description: 商业排序，内容强制比例，但是需要相关性足够高
    '''
    # 模板 中 设计师 类型值
    DISPLAY_DESIGNER_TYPE = 1

    # 设计师 展示 的 数量
    DISPLAY_DESIGNER_NUM = 2

    # 模板 中 非设计师 类型值
    DISPLAY_NOT_DESIGNER_TYPE = 2
    # 模板 中 非设计师(有符合要求的wiki) 类型值
    # DISPLAY_NOT_DESIGNER_WIKI_TYPE = 2
    # 模板 中 非设计师(没有符合要求的wiki) 类型值
    # DISPLAY_NOT_DESIGNER_NO_WIKI_TYPE = 3

    # 非设计师（有符合要求的wiki） 展示 的 数量
    DISPLAY_NOT_DESIGNER_WIKI_NUM = 2

    # 非设计师（没有符合要求的wiki） 展示 的 数量
    DISPLAY_NOT_DESIGNER_NUM = 4
    # DISPLAY_NOT_DESIGNER_NO_WIKI_NUM = 2


    # # 模板 中 内容关联wiki 类型值
    # DISPLAY_WIKI_TYPE = 1

    # # 内容关联wiki 展示 的 数量
    # DISPLAY_WIKI_NUM = 2

    # # 模板 中 非内容关联wiki 类型值
    # DISPLAY_NOT_WIKI_TYPE = 2

    # # 非内容关联wiki 展示 的 数量
    # DISPLAY_NOT_WIKI_NUM = 4

    DISPLAY_DESIGNER_NUM_NEW = 2

    DISPLAY_DESIGNER_TYPE_NEW = 1

    DISPLAY_WIKI_NUM_NEW = 2

    DISPLAY_WIKI_TYPE_NEW = 2

    DISPLAY_OTHER_NUM_NEW = 2

    DISPLAY_OTHER_TYPE_NEW = 3

    # 地区设计师
    DISPLAY_AREA_DESIGNER_TYPE = 1
    # 非地区设计师
    DISPLAY_NOT_AREA_DESIGNER_TYPE = 2

    @classmethod
    def ranking(cls, intent_list, sortedObjIds, hash_obj_user_type, hash_obj_id_jaccard_sim, hash_obj_related_wiki):
        if "商品" in intent_list or "找商品" in intent_list:
            return sortedObjIds
            # wiki_obj_list = []
            # not_wiki_obj_list = []

            # for sort_obj_id in sortedObjIds:
            #     # 文本 jaccard 需要满分
            #     if len(sort_obj_id) > 0 and float(hash_obj_id_jaccard_sim[sort_obj_id]) == 1.0:
            #         # 内容关联wiki 非内容关联wiki 分组
            #         if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_related_wiki and hash_obj_related_wiki[sort_obj_id] == 1:
            #             wiki_obj_list.append(sort_obj_id)
            #         else:
            #             not_wiki_obj_list.append(sort_obj_id)

            # template = []
            # # 内容 关联 wiki
            # template.extend([cls.DISPLAY_WIKI_TYPE for i in range(cls.DISPLAY_WIKI_NUM)])
            # # 内容 没有关联 wiki
            # template.extend([cls.DISPLAY_NOT_WIKI_TYPE for i in range(cls.DISPLAY_NOT_WIKI_NUM)])

            # return_data = []
            # have_sorted_ids = set()
            # while True:
            #     random.shuffle(template)

            #     for display_type in template:
            #         if display_type == cls.DISPLAY_WIKI_TYPE:
            #             if len(wiki_obj_list) == 0:
            #                 break
            #             obj_id = wiki_obj_list.pop(0)
            #         else:
            #             if len(not_wiki_obj_list) == 0:
            #                 break
            #             obj_id = not_wiki_obj_list.pop(0)

            #         have_sorted_ids.add(obj_id)
            #         return_data.append(obj_id)

            #     if len(wiki_obj_list) == 0 or len(not_wiki_obj_list) == 0:
            #         break

            # for obj_id in sortedObjIds:
            #     if obj_id not in have_sorted_ids:
            #         return_data.append(obj_id)
        else:
            #  非设计师内容的wiki和非wiki 控制比例
            # designer_obj_list = []
            # # not_designer_obj_list = []
            # not_designer_no_wiki_obj_list = []
            # not_designer_wiki_obj_list = []
            #
            # for sort_obj_id in sortedObjIds:
            #     # 文本 jaccard 需要满分
            #     if len(sort_obj_id) > 0 and float(hash_obj_id_jaccard_sim[sort_obj_id]) == 1.0:
            #         # 设计师内容 非设计师内容 分组
            #         if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_user_type and hash_obj_user_type[sort_obj_id] == Member.AUTH_USER_DESIGNER:
            #             designer_obj_list.append(sort_obj_id)
            #         else:  # 非设计师用户
            #             # not_designer_obj_list.append(sort_obj_id)
            #             # 内容关联wiki 非内容关联wiki 分组
            #             if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_related_wiki and hash_obj_related_wiki[sort_obj_id] == 1:
            #                 not_designer_wiki_obj_list.append(sort_obj_id)
            #             else:
            #                 not_designer_no_wiki_obj_list.append(sort_obj_id)
            #
            # # print("designer_obj_list:", designer_obj_list)
            # # print("not_designer_obj_list:", not_designer_obj_list)
            # template = []
            # # 设计师
            # template.extend([cls.DISPLAY_DESIGNER_TYPE for i in range(cls.DISPLAY_DESIGNER_NUM)])
            # # 非设计师
            # # template.extend([cls.DISPLAY_NOT_DESIGNER_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_NO_WIKI_NUM)])
            # # 内容 关联 wiki
            # template.extend([cls.DISPLAY_NOT_DESIGNER_WIKI_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_WIKI_NUM)])
            # # 内容 没有关联 wiki
            # template.extend([cls.DISPLAY_NOT_DESIGNER_NO_WIKI_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_NO_WIKI_NUM)])
            #
            # return_data = []
            # have_sorted_ids = set()
            # while True:
            #     random.shuffle(template)
            #
            #     for display_type in template:
            #         if display_type == cls.DISPLAY_DESIGNER_TYPE:
            #             if len(designer_obj_list) == 0:
            #                 break
            #             obj_id = designer_obj_list.pop(0)
            #         # 非设计师挂了合适的wiki
            #         elif display_type == cls.DISPLAY_NOT_DESIGNER_WIKI_TYPE:
            #             # 补位逻辑  当非设计师挂了合适的wiki 数据没有时  用 非设计师没有挂合适的wiki 数据补
            #             if len(not_designer_wiki_obj_list) == 0:
            #                 if len(not_designer_no_wiki_obj_list) == 0:
            #                     break
            #                 obj_id = not_designer_no_wiki_obj_list.pop(0)
            #             else:
            #                 obj_id = not_designer_wiki_obj_list.pop(0)
            #         else:  # 非设计师没有挂合适的wiki
            #             if len(not_designer_no_wiki_obj_list) == 0:
            #                 break
            #             obj_id = not_designer_no_wiki_obj_list.pop(0)
            #
            #         have_sorted_ids.add(obj_id)
            #         return_data.append(obj_id)
            #
            #     if len(designer_obj_list) == 0 or len(not_designer_no_wiki_obj_list) == 0:
            #         break

            # 非设计师内容的wiki和非wiki不控制比例
            designer_obj_list = []
            not_designer_obj_list = []

            for sort_obj_id in sortedObjIds:
                # 文本 jaccard 需要满分
                if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_id_jaccard_sim and float(hash_obj_id_jaccard_sim[sort_obj_id]) == 1.0:
                    # 设计师内容 非设计师内容 分组
                    if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_user_type and hash_obj_user_type[sort_obj_id] == Member.AUTH_USER_DESIGNER:
                        designer_obj_list.append(sort_obj_id)
                    else:
                        not_designer_obj_list.append(sort_obj_id)

            # print("designer_obj_list:", designer_obj_list)
            # print("not_designer_obj_list:", not_designer_obj_list)
            template = []
            # 设计师
            template.extend([cls.DISPLAY_DESIGNER_TYPE for i in range(cls.DISPLAY_DESIGNER_NUM)])
            # 非设计师
            template.extend([cls.DISPLAY_NOT_DESIGNER_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_NUM)])

            return_data = []
            have_sorted_ids = set()
            while True:
                random.shuffle(template)

                for display_type in template:
                    if display_type == cls.DISPLAY_DESIGNER_TYPE:
                        if len(designer_obj_list) == 0:
                            break
                        obj_id = designer_obj_list.pop(0)
                    else:
                        if len(not_designer_obj_list) == 0:
                            break
                        obj_id = not_designer_obj_list.pop(0)

                    have_sorted_ids.add(obj_id)
                    return_data.append(obj_id)

                if len(designer_obj_list) == 0 or len(not_designer_obj_list) == 0:
                    break

            for obj_id in sortedObjIds:
                if obj_id not in have_sorted_ids:
                    return_data.append(obj_id)

        return return_data



    @classmethod
    def ranking_new(cls, intent_list, sortedObjIds, sortedObjIds_with_wiki_pool, hash_obj_user_type,
                    hash_obj_id_jaccard_split_match_sim, hash_obj_related_wiki, hash_obj_designer_flow, hashObjCtr):
        # 非设计师内容的wiki和非wiki不控制比例
        designer_obj_list = []
        wiki_obj_list = []
        other_obj_list = []
        reset_obj_ids = []

        sortedObjIdsNum = len(sortedObjIds)
        is_rela_top_num = 100
        if sortedObjIdsNum < 100:
            is_rela_top_num = 1
        elif sortedObjIdsNum >= 100 and sortedObjIdsNum < 200:
            is_rela_top_num = 42
        elif sortedObjIdsNum >= 200 and sortedObjIdsNum < 300:
            is_rela_top_num = 60
        elif sortedObjIdsNum >= 300 and sortedObjIdsNum < 400:
            is_rela_top_num = 80

        # print("is_rela_top_num", is_rela_top_num)
        is_rela_top_obj_set = set(sortedObjIds[:is_rela_top_num])

        top_obj_ctr_scores = []
        for obj_id in is_rela_top_obj_set:
            if obj_id in hashObjCtr:
                top_obj_ctr_scores.append(hashObjCtr[obj_id])

        # print("np.percentile(top_obj_ctr_scores, 50)", np.percentile(top_obj_ctr_scores, 50))
        # print("np.percentile(top_obj_ctr_scores, 75)", np.percentile(top_obj_ctr_scores, 75))
        # print("np.percentile(top_obj_ctr_scores, 90)", np.percentile(top_obj_ctr_scores, 90))
        # print("np.mean(top_obj_ctr_scores)", np.mean(top_obj_ctr_scores))
        median_ctr = np.percentile(top_obj_ctr_scores, 50)

        for sort_obj_id in sortedObjIds_with_wiki_pool:
            # 文本 jaccard 需要满分
            if len(sort_obj_id) > 0 and float(hash_obj_id_jaccard_split_match_sim[sort_obj_id]) == 1.0 \
                    and sort_obj_id in is_rela_top_obj_set and sort_obj_id in hashObjCtr and hashObjCtr[sort_obj_id] >= median_ctr:
                # 设计师内容 非设计师内容 分组
                if sort_obj_id in hash_obj_related_wiki and hash_obj_related_wiki[sort_obj_id] == 1:
                    reset_obj_ids.append(sort_obj_id)
                    wiki_obj_list.append(sort_obj_id)
                elif sort_obj_id in hash_obj_user_type and hash_obj_user_type[
                    sort_obj_id] == Member.AUTH_USER_DESIGNER and sort_obj_id in hash_obj_designer_flow and hash_obj_designer_flow[sort_obj_id] != "d":
                    reset_obj_ids.append(sort_obj_id)
                    designer_obj_list.append(sort_obj_id)
                else:
                    reset_obj_ids.append(sort_obj_id)
                    other_obj_list.append(sort_obj_id)

        # print("designer_obj_list:", designer_obj_list)
        # print("wiki_obj_list:", wiki_obj_list)
        # print("other_obj_list:", other_obj_list)

        template = []
        # 设计师
        template.extend([cls.DISPLAY_DESIGNER_TYPE_NEW for i in range(cls.DISPLAY_DESIGNER_NUM_NEW)])
        # Wiki
        template.extend([cls.DISPLAY_WIKI_TYPE_NEW for i in range(cls.DISPLAY_WIKI_NUM_NEW)])
        # 其他
        template.extend([cls.DISPLAY_OTHER_TYPE_NEW for i in range(cls.DISPLAY_OTHER_NUM_NEW)])

        return_data = []
        have_sorted_ids = set()
        while True:
            random.shuffle(template)

            for display_type in template:
                wiki_obj_num = len(wiki_obj_list)
                designer_obj_num = len(designer_obj_list)

                if display_type == cls.DISPLAY_WIKI_TYPE_NEW:
                    if wiki_obj_num == 0 and designer_obj_num == 0:
                        break

                    if wiki_obj_num > 0:
                        obj_id = wiki_obj_list.pop(0)
                    else:
                        obj_id = designer_obj_list.pop(0)
                elif display_type == cls.DISPLAY_DESIGNER_TYPE_NEW:
                    if wiki_obj_num == 0 and designer_obj_num == 0:
                        break

                    if designer_obj_num > 0:
                        obj_id = designer_obj_list.pop(0)
                    else:
                        obj_id = wiki_obj_list.pop(0)
                else:
                    if len(other_obj_list) == 0:
                        break
                    obj_id = other_obj_list.pop(0)

                have_sorted_ids.add(obj_id)
                return_data.append(obj_id)

            if (len(designer_obj_list) == 0 and len(wiki_obj_list) == 0) or len(other_obj_list) == 0:
                break

        for obj_id in sortedObjIds:
            if obj_id not in have_sorted_ids:
                return_data.append(obj_id)

        return return_data, reset_obj_ids

    @classmethod
    def ranking_area_designer_new(cls, sortedObjIds, sortedObjIds_with_wiki_pool, hash_obj_user_type,
                              hash_obj_id_jaccard_split_match_sim,
                              hashObjCtr, hash_main_service_area, user_area):
        # 非设计师内容的wiki和非wiki不控制比例
        area_designer_obj_list_tmp = []
        not_area_designer_obj_list_tmp = []
        not_designer_obj_list = []
        designer_obj_list = []
        reset_obj_ids = []

        sortedObjIdsNum = len(sortedObjIds)
        is_rela_top_num = 100
        if sortedObjIdsNum < 100:
            is_rela_top_num = 1
        elif sortedObjIdsNum >= 100 and sortedObjIdsNum < 200:
            is_rela_top_num = 42
        elif sortedObjIdsNum >= 200 and sortedObjIdsNum < 300:
            is_rela_top_num = 60
        elif sortedObjIdsNum >= 300 and sortedObjIdsNum < 400:
            is_rela_top_num = 80

        is_rela_top_obj_set = set(sortedObjIds[:is_rela_top_num])

        top_obj_ctr_scores = []
        for obj_id in is_rela_top_obj_set:
            if obj_id in hashObjCtr:
                top_obj_ctr_scores.append(hashObjCtr[obj_id])

        median_ctr = 0
        if len(top_obj_ctr_scores) > 0:
            median_ctr = np.percentile(top_obj_ctr_scores, 50)

        for sort_obj_id in sortedObjIds_with_wiki_pool:
            # 文本 jaccard 需要满分
            if len(sort_obj_id) > 0 and float(hash_obj_id_jaccard_split_match_sim[sort_obj_id]) == 1.0 \
                    and sort_obj_id in is_rela_top_obj_set and sort_obj_id in hashObjCtr and hashObjCtr[
                sort_obj_id] >= median_ctr:
                # 设计师内容 非设计师内容 分组
                if len(sort_obj_id) == 0 and sort_obj_id in hash_obj_user_type:
                    continue

                if hash_obj_user_type[sort_obj_id] == Member.AUTH_USER_DESIGNER:
                    if len(area_designer_obj_list_tmp) < 6 and sort_obj_id in hash_main_service_area and \
                            hash_main_service_area[sort_obj_id] == user_area:
                        area_designer_obj_list_tmp.append(sort_obj_id)
                    else:
                        not_area_designer_obj_list_tmp.append(sort_obj_id)

                    designer_obj_list.append(sort_obj_id)
                    reset_obj_ids.append(sort_obj_id)
                else:
                    not_designer_obj_list.append(sort_obj_id)
                    reset_obj_ids.append(sort_obj_id)

        # print("designer_obj_list:", designer_obj_list)
        # print("not_designer_obj_list:", not_designer_obj_list)
        template = []
        # 设计师
        template.extend([cls.DISPLAY_DESIGNER_TYPE for i in range(cls.DISPLAY_DESIGNER_NUM)])
        # 非设计师
        template.extend([cls.DISPLAY_NOT_DESIGNER_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_NUM)])

        # 地区设计师 和 非地区设计师
        return_data = []
        have_sorted_ids = set()
        circleNum = 0
        head18_designer_obj_ids = set()

        area_designer_ratio_list = [cls.DISPLAY_AREA_DESIGNER_TYPE] * 6
        area_designer_ratio_list.extend([cls.DISPLAY_AREA_DESIGNER_TYPE] * 12)
        random.shuffle(area_designer_ratio_list)

        designer_obj_list_num = len(designer_obj_list)

        area_designer_obj_list = []
        not_area_designer_obj_list = []
        while True:
            random.shuffle(template)
            circleNum = circleNum + 1

            for display_type in template:
                if display_type == cls.DISPLAY_DESIGNER_TYPE:
                    if designer_obj_list_num == 0:
                        break

                    designer_obj_list_num = designer_obj_list_num - 1
                    # 前18条内容正常排序 不区分是否是 地区设计师
                    if circleNum <= 3:
                        if len(designer_obj_list) == 0:
                            break
                        obj_id = designer_obj_list.pop(0)
                        head18_designer_obj_ids.add(obj_id)

                        # 过滤已经推过的设计师内容
                        if circleNum == 3:
                            for item in area_designer_obj_list_tmp:
                                if item not in head18_designer_obj_ids:
                                    area_designer_obj_list.append(item)

                            for item in not_area_designer_obj_list_tmp:
                                if item not in head18_designer_obj_ids:
                                    not_area_designer_obj_list.append(item)

                            area_designer_ratio_list = [cls.DISPLAY_AREA_DESIGNER_TYPE] * len(area_designer_obj_list)

                            size = len(area_designer_obj_list) * 2
                            if len(not_area_designer_obj_list) < size:
                                size = len(not_area_designer_obj_list)
                            area_designer_ratio_list.extend([cls.DISPLAY_NOT_AREA_DESIGNER_TYPE] * size)

                            random.shuffle(area_designer_ratio_list)

                    else:
                        # 随机展示 地区 还是 非地区 设计师
                        display_designer_type = cls.DISPLAY_NOT_AREA_DESIGNER_TYPE
                        if len(area_designer_ratio_list) > 0:
                            display_designer_type = area_designer_ratio_list.pop(0)

                        # 地区不够  非地区来凑
                        if display_designer_type == cls.DISPLAY_AREA_DESIGNER_TYPE and len(area_designer_obj_list) > 0:
                            obj_id = area_designer_obj_list.pop(0)
                        else:
                            obj_id = not_area_designer_obj_list.pop(0)
                else:
                    if len(not_designer_obj_list) == 0:
                        break
                    obj_id = not_designer_obj_list.pop(0)

                have_sorted_ids.add(obj_id)
                return_data.append(obj_id)

            if designer_obj_list_num == 0 or len(not_designer_obj_list) == 0:
                break

        for obj_id in sortedObjIds:
            if obj_id not in have_sorted_ids:
                return_data.append(obj_id)

        return return_data, reset_obj_ids

    @classmethod
    def ranking_area_designer(cls, sortedObjIds, hash_obj_user_type, hash_obj_id_jaccard_sim, hash_main_service_area, user_area):
        # 非设计师内容的wiki和非wiki不控制比例
        area_designer_obj_list_tmp = []
        not_area_designer_obj_list_tmp = []
        not_designer_obj_list = []
        designer_obj_list = []

        for sort_obj_id in sortedObjIds:
            # 文本 jaccard 需要满分
            if len(sort_obj_id) > 0 and sort_obj_id in hash_obj_id_jaccard_sim and float(hash_obj_id_jaccard_sim[sort_obj_id]) == 1.0:
                # 设计师内容 非设计师内容 分组
                if len(sort_obj_id) == 0 and sort_obj_id in hash_obj_user_type:
                    continue

                if hash_obj_user_type[sort_obj_id] == Member.AUTH_USER_DESIGNER:
                    if len(area_designer_obj_list_tmp) < 6 and sort_obj_id in hash_main_service_area and hash_main_service_area[sort_obj_id] == user_area:
                        area_designer_obj_list_tmp.append(sort_obj_id)
                    else:
                        not_area_designer_obj_list_tmp.append(sort_obj_id)

                    designer_obj_list.append(sort_obj_id)
                else:
                    not_designer_obj_list.append(sort_obj_id)

        # print("designer_obj_list:", designer_obj_list)
        # print("not_designer_obj_list:", not_designer_obj_list)
        template = []
        # 设计师
        template.extend([cls.DISPLAY_DESIGNER_TYPE for i in range(cls.DISPLAY_DESIGNER_NUM)])
        # 非设计师
        template.extend([cls.DISPLAY_NOT_DESIGNER_TYPE for i in range(cls.DISPLAY_NOT_DESIGNER_NUM)])

        # 地区设计师 和 非地区设计师
        return_data = []
        have_sorted_ids = set()
        circleNum = 0
        head18_designer_obj_ids = set()

        area_designer_ratio_list = [cls.DISPLAY_AREA_DESIGNER_TYPE] * 6
        area_designer_ratio_list.extend([cls.DISPLAY_AREA_DESIGNER_TYPE] * 12)
        random.shuffle(area_designer_ratio_list)

        designer_obj_list_num = len(designer_obj_list)

        area_designer_obj_list = []
        not_area_designer_obj_list = []
        while True:
            random.shuffle(template)
            circleNum = circleNum + 1

            for display_type in template:
                if display_type == cls.DISPLAY_DESIGNER_TYPE:
                    if designer_obj_list_num == 0:
                        break

                    designer_obj_list_num = designer_obj_list_num - 1
                    # 前18条内容正常排序 不区分是否是 地区设计师
                    if circleNum <= 3:
                        if len(designer_obj_list) == 0:
                            break
                        obj_id = designer_obj_list.pop(0)
                        head18_designer_obj_ids.add(obj_id)

                        # 过滤已经推过的设计师内容
                        if circleNum == 3:
                            for item in area_designer_obj_list_tmp:
                                if item not in head18_designer_obj_ids:
                                    area_designer_obj_list.append(item)

                            for item in not_area_designer_obj_list_tmp:
                                if item not in head18_designer_obj_ids:
                                    not_area_designer_obj_list.append(item)

                            area_designer_ratio_list = [cls.DISPLAY_AREA_DESIGNER_TYPE] * len(area_designer_obj_list)

                            size = len(area_designer_obj_list) * 2
                            if len(not_area_designer_obj_list) < size:
                                size = len(not_area_designer_obj_list)
                            area_designer_ratio_list.extend([cls.DISPLAY_NOT_AREA_DESIGNER_TYPE] * size)

                            random.shuffle(area_designer_ratio_list)

                    else:
                        # 随机展示 地区 还是 非地区 设计师
                        display_designer_type = cls.DISPLAY_NOT_AREA_DESIGNER_TYPE
                        if len(area_designer_ratio_list) > 0:
                            display_designer_type = area_designer_ratio_list.pop(0)

                        # 地区不够  非地区来凑
                        if display_designer_type == cls.DISPLAY_AREA_DESIGNER_TYPE and len(area_designer_obj_list) > 0:
                            obj_id = area_designer_obj_list.pop(0)
                        else:
                            obj_id = not_area_designer_obj_list.pop(0)
                else:
                    if len(not_designer_obj_list) == 0:
                        break
                    obj_id = not_designer_obj_list.pop(0)

                have_sorted_ids.add(obj_id)
                return_data.append(obj_id)

            if designer_obj_list_num == 0 or len(not_designer_obj_list) == 0:
                break

        # 对处理好的数据 每一个模板内 按照 相似度  从高到低排

        for obj_id in sortedObjIds:
            if obj_id not in have_sorted_ids:
                return_data.append(obj_id)

        return return_data


















