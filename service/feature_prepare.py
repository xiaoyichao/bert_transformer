import time
import datetime
import os
from common.common_log import Logger, get_log_name
from service.Init.InitService import InitService
from data_utils.click_data4bert import get_tf_web
from bert_recall.convert_dssm_to_tfrecord_v2 import FLAGS

# 删除重复定义
if "vocab_file" in FLAGS:
  FLAGS.__delattr__("vocab_file")

if "use_pkl" in FLAGS:
  FLAGS.__delattr__("use_pkl")

from bert_regress.predict4tf_serving_user_profile_v2 import get_examples as user_profile_get_examples, \
    convert_examples_to_features as user_profile_convert_examples_to_features, \
    tokenizer as user_profile_tokenizer
from common.tool import Tool


abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class feature_prepare:

    @classmethod
    def prepare_feature(cls, uid, query, query_splited_list, user_profile_data, user_click_seq_historys,
                        user_favorite_seq_historys, sortedObjIds, hashEsDataInfos, ab_test, unique_str):
        # 特征构造
        starttime = time.time()

        def get_delta_days(yesterday, today):
            dt2 = datetime.datetime.fromtimestamp(today)
            dt2 = dt2.replace(hour=0, minute=0, second=0, microsecond=0)
            dt1 = datetime.datetime.fromtimestamp(yesterday)
            dt1 = dt1.replace(hour=0, minute=0, second=0, microsecond=0)
            return (dt2 - dt1).days

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        wiki_dict = {1: "是", 0: "否"}

        # 用户类型
        user_identity_type_dict = {
            "user_common": "普通用户",
            "auth_user_brand": "品牌用户",
            "auth_user_designer": "设计师用户",
            "auth_user_v": "个人V认证用户",
            "auth_user_unauth_designer": "未认证设计师用户",
            "auth_org_v": "机构V认证用户",
            "auth_deco_company": "装修公司用户"
        }

        # 设计师意向标签
        designer_intention_group_tag_dict = {
            "前10%(含)": True,
            "40%～50%(含)": True,
            "30%～40%(含)": True,
            "20%～30%(含)": True,
            "60%～70%(含)": True,
            "80%～90%(含)": True,
            "50%～60%(含)": True,
            "90%～100%(含)": True,
            "10%～20%(含)": True,
            "70%～80%(含)": True,
        }

        # 性别
        gender_dict = {
            1: "男",
            2: "女",
        }

        # 房屋状态
        home_status_dict = {
            "毛坯房": True,
            "二手房": True,
            "精装修": True,
        }

        # 装修状态
        decoration_status_dict = {
            "准备装修": True,
            "正在装修": True,
            "不需要装修": True,
        }

        # 用户类型 aab
        user_identity_type = "未知用户类型"
        if "aab" in user_profile_data:
            user_profile_user_type = user_profile_data["aab"].strip()
            user_identity_type = "未知用户类型" if user_profile_user_type not in user_identity_type_dict.keys() else \
                user_identity_type_dict[user_profile_user_type]

        # 设计师意向标签  aap
        designer_intention_group_tag = "未知设计师意向类型"
        if "aap" in user_profile_data:
            designer_intention_group_tag = user_profile_data["aap"].strip()
            designer_intention_group_tag = "未知设计师意向类型" if designer_intention_group_tag not in designer_intention_group_tag_dict.keys() else designer_intention_group_tag

        # 性别 adb
        gender = "未知性别类型"
        if "adb" in user_profile_data:
            gender = int(user_profile_data["adb"])
            gender = "未知性别类型" if gender not in gender_dict.keys() else gender_dict[gender]

        # 房屋装修状态 abr
        home_status = "未知房屋状态类型"
        if "abr" in user_profile_data:
            home_status = user_profile_data["abr"].strip()
            home_status = "未知房屋状态类型" if home_status not in home_status_dict.keys() else home_status

        # 装修状态 acy
        decoration_status = "未知装修状态类型"
        if "acy" in user_profile_data:
            decoration_status = user_profile_data["acy"].strip()
            decoration_status = "未知装修状态类型" if decoration_status not in decoration_status_dict.keys() else decoration_status

        # 找商品意向分数 acx
        user_wiki_intention_score = 0
        if "acx" in user_profile_data:
            user_wiki_intention_score = float(user_profile_data["acx"])

        # 位置信息
        position = 1

        # age(暂时缺失)
        age = 0
        if "afb" in user_profile_data:
            age = float(user_profile_data["afb"])

        # home_space (房屋面积)
        home_space = 0
        if "add" in user_profile_data:
            home_space = float(user_profile_data["add"])

        # house_room_count(户型居室) abs
        house_room_count = 0
        if "abs" in user_profile_data:
            house_room_count = float(user_profile_data["abs"])

        # budget 预算 adj
        budget = 0
        if "adj" in user_profile_data:
            budget = float(user_profile_data["adj"])

        # having_intention_for_decorate_company acz
        having_intention_for_decorate_company = 0
        if "acz" in user_profile_data:
            having_intention_for_decorate_company = float(user_profile_data["acz"])

        # 用户有效点击序列
        transform_user_click_seq_historys = []
        if len(user_click_seq_historys) > 0:
            for user_click_seq_history in user_click_seq_historys:
                if user_click_seq_history in InitService.objId2IndexForUserProfile:
                    objId2Index = InitService.objId2IndexForUserProfile[user_click_seq_history]
                    transform_user_click_seq_historys.append(objId2Index)

        # 用户收藏序列
        transform_user_favorite_seq_historys = []
        if len(user_favorite_seq_historys) > 0:
            for user_favorite_seq_history in user_favorite_seq_historys:
                if user_favorite_seq_history in InitService.objId2IndexForUserProfile:
                    objId2Index = InitService.objId2IndexForUserProfile[user_favorite_seq_history]
                    transform_user_favorite_seq_historys.append(objId2Index)

        lines = []

        for obj_id in sortedObjIds:
            keys = hashEsDataInfos[obj_id].keys()

            if "title" in keys:
                title = hashEsDataInfos[obj_id]["title"]
            else:
                title = ""

            if "desc" in keys:
                remark = hashEsDataInfos[obj_id]["desc"]
            else:
                remark = ""

            if "favorite" in keys:
                favorite_num = float(hashEsDataInfos[obj_id]["favorite"])
            else:
                favorite_num = float(0)

            if "comment" in keys:
                comment_num = float(hashEsDataInfos[obj_id]["comment"])
            else:
                comment_num = float(0)

            if "like" in keys:
                like_num = float(hashEsDataInfos[obj_id]["like"])
            else:
                like_num = float(0)

            if "admin_score" in keys:
                score = float(hashEsDataInfos[obj_id]["admin_score"])
            else:
                score = float(0)

            if "publish_time" in keys:
                publish_time = int(hashEsDataInfos[obj_id]["publish_time"])
            else:
                publish_time = int(time.time())

            if "is_relate_wiki" in keys:
                wiki = int(hashEsDataInfos[obj_id]["is_relate_wiki"])
            else:
                wiki = 0

            if "split_words_title_array" in keys:
                split_words_for_ai_title_list = hashEsDataInfos[obj_id]["split_words_title_array"]
            else:
                split_words_for_ai_title_list = []

            if "split_words_remark_array" in keys:
                split_words_for_ai_remark_list = hashEsDataInfos[obj_id]["split_words_remark_array"]
            else:
                split_words_for_ai_remark_list = []

            obj_type_num = Tool.getTypeByObjId(obj_id)
            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict.keys() else obj_type_dict[obj_type_num]
            wiki = "未知wiki类型" if wiki not in wiki_dict.keys() else wiki_dict[wiki]

            if obj_type == "note" or obj_type == "回答的note":  # note分数转化，30转化为60
                if score < 60:
                    score = 60

            today = int(time.time())
            interval_days = get_delta_days(publish_time, today)

            title_tf_num = get_tf_web(query, query_splited_list, split_words_for_ai_title_list, have_jiebad=True, max_len=64,
                                      isGetUnique = False)
            remark_tf_num = get_tf_web(query, query_splited_list, split_words_for_ai_remark_list, have_jiebad=True, max_len=64,
                                       isGetUnique = False)

            # 待预测 obj id
            targetObj2index = 0
            if obj_id in InitService.objId2IndexForUserProfile:
                targetObj2index = InitService.objId2IndexForUserProfile[obj_id]

            line = [query, title, remark, '-1', title_tf_num, remark_tf_num, favorite_num, like_num,
                    comment_num, score, interval_days, wiki, obj_type, user_identity_type,
                    designer_intention_group_tag,
                    user_wiki_intention_score, position, gender, age, home_status, home_space, house_room_count,
                    budget,
                    decoration_status, having_intention_for_decorate_company, targetObj2index,
                    transform_user_click_seq_historys, transform_user_favorite_seq_historys]

            lines.append(line)

        # 数据写入对象
        examples = user_profile_get_examples(lines)

        # 数据id化
        max_seq_length = 64

        features = user_profile_convert_examples_to_features(examples, max_seq_length, user_profile_tokenizer)

        endtime = time.time()

        log.logger.info("构建 特征数据 为 用户个性化搜索 的 运行时间：{:.10f} s".format(
            endtime - starttime) + " uid:" + uid + " query:" + query + " unique_str:" + unique_str)

        return features