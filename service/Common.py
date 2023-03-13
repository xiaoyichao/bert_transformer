# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/26 23:23
@Desciption : 
'''
import time
import random
import configparser
import codecs
import requests
import numpy as np
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")
import grpc
from term.term_weight import TermWeight
from cache.SearchContent import SearchContent as cSearchContent
os.environ['TF_KERAS'] = '1'
from keras_bert import Tokenizer
from rw_vector.vecbin4recall import VecBin
from common.common_log import Logger, get_log_name
from service.Init.InitService import InitService
from db.hhzTag.TagPackage import TagPackage
from service.Tool.Synonym import Synonym
from service.Tool.SynonymTest import SynonymTest
from db.hhzMember.Member import Member as dbMember
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)
use_RAM_cache = False if search_opt_common_config["RAM_cache"]["use_RAM_cache"] == "False" else True

common_config = configparser.ConfigParser()
common_config.read("/home/resource_config/search_opt/common.ini")

bert_server_url = common_config["bert_server_url"]["url"]

token_dict = {}
with codecs.open("/home/adm_rsync_dir/search_op/vocab.txt", "r", "utf-8") as fr:
    for line in fr:
        line = line.strip()
        token_dict[line] = len(token_dict)

tokenizer = Tokenizer(token_dict)

class Common(object):
    # 搜索方式 热度 字符串
    SEARCH_TYPE_HOT_STR = "hot"

    # 搜索方式 时间 字符串
    SEARCH_TYPE_TIME_STR = "time"

    # 过滤的uid
    FILTER_SEARCH_UIDS = [
        5068780, 4684158, 1193321, 378994, 22829, 11953, 11941, 10001
    ]

    # 普通内容的分数  note 普通为30 转换为60
    NORMAL_ADMIN_SCORE = 60

    PHOTO_TYPE = "photo"
    PHOTO_TYPE_NUMBER = 0

    ARTICLE_TYPE = "article"

    BLANK_TYPE = "blank"

    ANSWER_TYPE = "answer"

    VIDEO_TYPE = "video"

    DIARY_TYPE = "diary"

    # 内容类型
    ALL_CONTENT_TYPE = [
        PHOTO_TYPE,
        ARTICLE_TYPE,
        BLANK_TYPE,
        ANSWER_TYPE,
        VIDEO_TYPE,
    ]

    # 内容是非住友类型
    IS_NOT_OWNER_TYPE = [
        dbMember.AUTH_USER_BRAND,
        dbMember.AUTH_USER_DESIGNER,
        dbMember.AUTH_USER_V,
        dbMember.AUTH_USER_UNAUTH_DESIGNER,
        dbMember.AUTH_ORG_V,
        dbMember.AUTH_DECO_COMPANY,
    ]

    # total article 内容类型

    # 整屋类型
    TYPE_TOTAL_ARTICLE_ARTICLE = 1

    # 指南类型
    TYPE_TOTAL_ARTICLE_GUIDE = 2

    # 经验文章 类型
    TYPE_TOTAL_ARTICLE_BLANK = 5

    # 搜索类型
    # 热门搜索
    SEARCH_TYPE_HOT = 1
    # 时间搜索
    SEARCH_TYPE_TIME = 2

    # 一页展示的数据
    PAGESIZE = 20

    # 需要走意图查询的 意图
    needRunIntentClass = [
        "装修知识",
        "找服务",
        "商品",
    ]

    INDEX_SEARCH_OP_PHOTO = "search_op_photo"

    INDEX_SEARCH_OP_TOTAL_ARTICLE = "search_op_total_article"

    # 设计师流量等级
    DESIGNER_A_FLOW_LEVEL = "dfp_a"

    DESIGNER_B_FLOW_LEVEL = "dfp_b"

    DESIGNER_C_FLOW_LEVEL = "dfp_c"

    @classmethod
    def getKeywordVect(self, init_resource_cls, keyword, init_request_global_var):
        """
        获取关键词 bert向量
        Args:
            keyword: 搜索词
        Returns:

        """
        starttime = time.time()
        if use_RAM_cache:
            query_vec = cSearchContent.get_query_vect(init_resource_cls, keyword)
        else:
            query_vec = None

        if query_vec is None:
            url = search_opt_common_config["bert_server_url"]["url"] + '/sentence_bert'
            data = {'sentence': keyword}
            r = requests.post(url, data)
            KeywordVectResult = eval(r.text)

            query_vec = np.array([KeywordVectResult["data"]["sentences_vec"]])
            # 归一化
            vecBinCls = VecBin()
            query_vec = vecBinCls.normalize(query_vec)

            cSearchContent.add_query_vect(init_resource_cls, keyword, str(query_vec.tolist()))

        endtime = time.time()
        log.logger.info("get_bert运行时间：{:.10f} s".format(endtime - starttime) + " unique_str:" + init_request_global_var.UNIQUE_STR)

        return np.array(query_vec, dtype="float32")

    @classmethod
    def getKeywordNewVect(self, init_resource_cls, keyword, unique_str):
        """
        获取 新的 关键词 bert向量
        Args:
            keyword: 搜索词
        Returns:

        """
        starttime = time.time()
        if use_RAM_cache:
            query_vec = cSearchContent.get_new_query_vect(init_resource_cls, keyword)
        else:
            query_vec = None

        if query_vec is None:
            channel = grpc.insecure_channel(str(common_config["vec_recall"]["tf_serving_rpc_url"]) + ":" + str(common_config["vec_recall"]["tf_serving_rpc_port"]))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

            max_len = 64
            qt, qs = tokenizer.encode(keyword, max_len=max_len)

            # 生成request
            request = predict_pb2.PredictRequest()
            request.model_spec.signature_name = ''  # set in estimator output
            request.model_spec.name = 'bert_dssm_query'
            # request.model_spec.version.value = 'v1'
            request.inputs['query_input_token'].CopyFrom(
                tf.make_tensor_proto([qt], dtype=tf.float32)
            )
            request.inputs['query_input_segment'].CopyFrom(
                tf.make_tensor_proto([qs], dtype=tf.float32)
            )

            # resp = stub.Predict.future(request, timeout=1)
            resp = stub.Predict.future(request)


            res = resp.result().outputs

            query_vec = np.squeeze(tf.make_ndarray(res['query_sem'])).tolist()

            cSearchContent.add_new_query_vect(init_resource_cls, keyword, str(query_vec))

        endtime = time.time()
        log.logger.info("getKeywordNewVect运行时间：{:.10f} s".format(endtime - starttime) + " unique_str:" + unique_str)

        return np.array(query_vec, dtype="float32")

    @classmethod
    def get_intent_info(self, init_resource_cls, keyword, query_vec):
        """
        获取 搜索词 意图
        Args:
            keyword: 搜索词
            query_vec: 搜索词 向量
        Returns:

        """
        if use_RAM_cache:
            query_intent_list = cSearchContent.get_query_intent(init_resource_cls, keyword)
        else:
            query_intent_list = None

        if query_intent_list is None:
            query_intent_list_tmp = InitService.IntentModel.predict_intent(keyword, query_vec)

            query_intent_list = set()
            for query_intent in query_intent_list_tmp:
                if query_intent == "找商品":
                    query_intent = "商品"

                query_intent_list.add(query_intent)

            cSearchContent.add_query_intent(init_resource_cls, keyword, str(list(query_intent_list)))

        return query_intent_list

    @classmethod
    def get_search_filter_tags(self, init_resource_cls):
        """
        获取 搜索过滤 标签
        Args:
        Returns:

        """
        search_filter_tags = cSearchContent.get_search_filter_tags(init_resource_cls)

        if search_filter_tags is None:
            search_filter_tags = TagPackage.get_search_filter_tags(init_resource_cls)
            cSearchContent.add_search_filter_tags(init_resource_cls, str(search_filter_tags))

        return search_filter_tags

    @classmethod
    def get_synonym_words(self, beSynonymWords):
        """
        获取 搜索词 的 同义词
        Args:
            beSynonymWords:  待获取同义词的列表

        Returns: set 同义词

        """
        synonym_word_map = {}

        for beSynonymWord in beSynonymWords:
            synonym_words = list(Synonym.GetSynonymByKeywords([beSynonymWord]))
            if len(synonym_words) > 6:
                synonym_words = random.sample(synonym_words, 6)

            if len(synonym_words) > 0:
                synonym_word_map[beSynonymWord] = synonym_words

        return synonym_word_map

    @classmethod
    def get_synonym_words_test(self, beSynonymWords):
        """
        获取 搜索词 的 同义词
        Args:
            beSynonymWords:  待获取同义词的列表

        Returns: set 同义词

        """
        synonym_word_map = {}

        for beSynonymWord in beSynonymWords:
            synonym_words = list(SynonymTest.GetSynonymByKeywords([beSynonymWord]))
            if len(synonym_words) > 5:
                synonym_words = random.sample(synonym_words, 5)

            if len(synonym_words) > 0:
                synonym_word_map[beSynonymWord] = synonym_words

        return synonym_word_map

    @classmethod
    def get_use_content_index(self, content_types_set):
        """
        根据 内容筛选 获取 需要到哪些内容索引获取数据
        Args:
            keyword: 搜索词
            queryVec:  搜索词向量
            is_owner: 是否是住友发布
            content_types: 内容类型

        Returns:

        """
        is_use_note_index = False
        use_note_type = [Common.PHOTO_TYPE, Common.ANSWER_TYPE, Common.VIDEO_TYPE]
        use_note_types_set = set(use_note_type)
        if len(use_note_types_set & content_types_set) > 0:
            is_use_note_index = True

        is_use_total_article_index = False
        use_total_article_type = [Common.ARTICLE_TYPE, Common.BLANK_TYPE]
        use_total_article_type_set = set(use_total_article_type)

        if len(use_total_article_type_set & content_types_set) > 0:
            is_use_total_article_index = True

        return is_use_note_index, is_use_total_article_index

    @classmethod
    def insertContent(self, insertObjId, objIds, index):
        """
        将 内容 插入 内容列表 的 某个位置
        Args:
            insertObjId: 待插入的内容id
            objIds:  内容列表
            index: 位置信息
        Returns: list 内容列表

        """
        result = objIds

        if len(objIds) > 0 and len(insertObjId) > 0:
            mergeData = []

            frontInfos = objIds[0:index]
            behindInfos = objIds[index:]

            mergeData.extend(frontInfos)
            mergeData.append(insertObjId)
            mergeData.extend(behindInfos)

            if len(mergeData) > 0:
                result = mergeData

        return result

    @classmethod
    def get_core_and_limit_words(cls, terms_weight, synonym_map):
        """
        将词权重 划分为 核心词 和 限定词 包括同义词
        Args:
            terms_weight: 待插入的内容id
            synonym_map:  内容列表
        Returns:

        """
        core_words = set()
        limit_words = set()

        core_synonym_words = set()
        limit_synonym_words = set()

        if len(terms_weight) > 0:
            for keyword, term_type in terms_weight.items():
                if str(term_type) == TermWeight.core_word:
                    core_words.add(keyword)
                    if keyword in synonym_map and len(synonym_map[keyword]) > 0:
                        core_synonym_words = core_synonym_words | set(synonym_map[keyword])

                if str(term_type) == TermWeight.limit_word:
                    limit_words.add(keyword)
                    if keyword in synonym_map and len(synonym_map[keyword]) > 0:
                        limit_synonym_words = limit_synonym_words | set(synonym_map[keyword])

        return core_words, limit_words, core_synonym_words, limit_synonym_words

    @classmethod
    def get_uid_by_obj_id(cls, obj_id):
        """
        从obj id 中获取 uid
        Args:
            obj_id: 内容id
        Returns:

        """
        if len(obj_id) == 0:
            return 0

        def trans_map(cint):
            if cint < 0:
                print("不合法")
                return
            elif cint < 10:
                return cint

            elif cint >= 10:
                return chr(cint - 10 + 65)

        # 将一个m进制的数转换为一个n进制的数
        def transfer(m, n, origin):
            num = anyToTen(m, origin)
            target = tenToAny(n, num)
            return target

        def anyToTen(m, origin):
            # 任意进制的数转换为10机制
            # 先将m转换为10进制
            # 公式 num = an * m**(n-1) + an-1 * m**(n-2).....+ a0 * m**0
            # 直接利用int的自带功能
            return int(str(origin), base=m)

        def tenToAny(n, origin):
            # 10进制转换为任意进制的数
            list = []
            while True:
                # 取商
                s = origin // n
                # 取余数
                tmp = origin % n
                list.append(trans_map(tmp))
                if s == 0:
                    break
                origin = s
            list.reverse()
            list = [str(each) for each in list]
            return ''.join(list)

        return int(transfer(36, 10, obj_id[-7:]))

    @classmethod
    def break_up_data_by_uid(cls, sorted_ids):
        """
        将内容打散，防止相同用户的内容出现的同一屏
        Args:
            sorted_ids: 排序好的内容id 列表
        Returns:

        """
        # 不重复uid间隔的数量
        interval_num = 20

        return_ids = []
        if len(sorted_ids) > 0:
            # 前15条内容的uid
            front_15_uids_set = []
            # 重复uid的内容 待被插入
            be_wait_insert_ids = []

            # 当 重复uid内容的id 和 排序好的id 都被分配完了
            while not (len(be_wait_insert_ids) == 0 and len(sorted_ids) == 0):
                # 分配 重复uid的内容id
                if len(be_wait_insert_ids) > 0:
                    # 重复uid内容的id列表中 被分配过的索引 最后删除重复uid列表中的数据
                    need_del_index = []

                    # 循环被重复的id列表
                    for index in range(len(be_wait_insert_ids)):
                        sorted_id = be_wait_insert_ids[index]

                        if len(sorted_id) > 0:
                            # 内容id 获取uid
                            uid = int(cls.get_uid_by_obj_id(sorted_id))

                            # 排序好的内容id列表为空时 直接将重复uid内容的id列表 放到返回列表的最后
                            if len(sorted_ids) == 0:
                                return_ids.extend(be_wait_insert_ids)
                                break
                            elif len(front_15_uids_set) >= interval_num:
                                # 存放前15条内容的uid列表 满了情况
                                if uid not in front_15_uids_set:
                                    # 存放待删除的索引
                                    need_del_index.append(index)
                                    # 删除 存放前15条内容中的第一条uid 并追加当前内容的uid
                                    front_15_uids_set.pop(0)
                                    front_15_uids_set.append(uid)
                                    # 添加返回内容id
                                    return_ids.append(sorted_id)
                            else:
                                # 存放前15条内容的uid列表 未满情况
                                if uid not in front_15_uids_set:
                                    need_del_index.append(index)
                                    front_15_uids_set.append(uid)
                                    return_ids.append(sorted_id)

                    be_wait_insert_ids_tmp = []
                    # 删除掉 待分配 的 重复列表中的数据
                    for be_wait_insert_id_index, be_wait_insert_id in enumerate(be_wait_insert_ids):
                        if be_wait_insert_id_index not in need_del_index:
                            be_wait_insert_ids_tmp.append(be_wait_insert_id)

                    be_wait_insert_ids = be_wait_insert_ids_tmp

                # 分配 原始排序好的 id
                if len(sorted_ids) > 0:
                    sorted_id = sorted_ids.pop(0)

                    if len(sorted_id) > 0:
                        uid = int(cls.get_uid_by_obj_id(sorted_id))

                        # 排序好的内容都被分配完了
                        if len(sorted_ids) == 0:
                            be_wait_insert_ids.insert(0, sorted_id)
                            return_ids.extend(be_wait_insert_ids)
                            be_wait_insert_ids = []
                        elif len(front_15_uids_set) >= interval_num:
                            # 存放前15条内容的uid列表 满了情况
                            # uid重复 放入重复uid列表
                            if uid in front_15_uids_set:
                                be_wait_insert_ids.append(sorted_id)
                            else:
                                # uid不重复 放入返回列表 并将uid加入前15条uid列表内容中
                                front_15_uids_set.pop(0)
                                front_15_uids_set.append(uid)
                                return_ids.append(sorted_id)
                        else:
                            # 存放前15条内容的uid列表 未满情况
                            # uid重复 放入重复uid列表
                            if uid in front_15_uids_set:
                                be_wait_insert_ids.append(sorted_id)
                            else:
                                # uid不重复 放入返回列表 并将uid加入前15条uid列表内容中
                                front_15_uids_set.append(uid)
                                return_ids.append(sorted_id)

        return return_ids

    @classmethod
    def is_obj_id_note(cls, obj_id):
        """
            obj_id: 内容id
            判断内容id是否是 note 类型
        """
        if obj_id[8] == '0' or obj_id[8] == '4':
            return True
        else:
            return False

    @classmethod
    def get_split_data(cls, keyword, is_get_unique_words = 1, use_split_test = 1, sep = ","):
        split_word_str = ""
        split_word_list = []

        headers = {
            'Accept': 'text/plain;charset=utf-8',
            'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
        }

        data = {"original_query": keyword,
                "unique_str": "111",
                "is_need_term": 0,
                "is_get_unique_words": is_get_unique_words,
                "use_split_test": use_split_test,
                }

        response = requests.post(
            bert_server_url + '/Cut_word_Term_weight', headers=headers, data=data)

        result = eval(response.content)

        if 'code' in result and result['code'] == 1:
            split_word_list = result["data"]["query_list"]
            if len(split_word_list) > 0:
                split_word_str = (sep +"").join(split_word_list)

        return split_word_str, split_word_list


if __name__ == '__main__':
    print(Common.is_obj_id_note("0003x7m0500664gn"))






