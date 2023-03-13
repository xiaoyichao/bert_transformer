# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/5 14:26
@Desciption :  初始化服务
'''
import configparser
import datetime
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../..")
from service.Tool.JiebaThread import JiebaThread
from service.Tool.SynonymThread import SynonymThread
from service.Tool.JiebaHHZ import JiebaHHZ
from service.Tool.Synonym import Synonym
from common.common_log import Logger, get_log_name
from service.faiss_map.ObjPassageVects import ObjPassageVects
from bert_regress import common4bert
from service.faiss_map.ObjVects import ObjVects
from intent_class.predict_intent import IntentClassModel
from common.get_ip import get_host_ip


# from ann_engine.faiss_engine4recall import SearchANN
from ann_engine.faiss_engine4newrecall import SearchANN
from common.create_connection import presto_execute
from service.faiss_map.FaissMap import FaissMap
from service.Tag import Tag
from cache.SearchContent import SearchContent
from db.hhzStore.brand import Brand
from common.tool import Tool
from db.hhzSearch.ObjId2Index import ObjId2Index
from cache.SeasonContent import SeasonContent
# import fasttext

import fasttext

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)


dir_name = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config_file = os.path.join(dir_name, '../../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)

svm_model_dir = "/data/search_opt_model/intent_model/SVM_model.m"
intent_pos_dir = "/data/search_opt_model/intent_model/intent_pos.txt"


class InitService(object):
    IntentModel = None

    SearchANN = None

    top_and_community_tag = None

    community_tag = None

    all_brand_name = set()

    item_model = None

    hash_city_designer_ratio = {}

    #  用于用户个性化搜索
    objId2IndexForUserProfile = {}

    @classmethod
    def initFeatureHandleService(cls, init_resource_cls):
        # 初始化 内容id 对应 唯一 数字 用户 用户行为序列
        cls.initObjId2Index(init_resource_cls)

    # 初始化 向量召回 服务
    @classmethod
    def initVecRecallService(cls):
        # 检查 dssm query pb 是否已使用最新的  gray 环境使用
        if get_host_ip() != "172.17.39.101":
            cls.checkQueryPbIsNewest()

        # 初始化 Ann 向量召回模型
        cls.InitAnnModel()

    @classmethod
    def initSearchOptService(cls, init_resource_cls, app):
        # 初始化 意图模型
        # cls.InitIntentClass()

        # 初始化 词权重的后台数据
        cls.init_term_weight_backstage(init_resource_cls)

        # 获取 top query
        cls.get_top_query(init_resource_cls)

        # 获取 品牌账号 和 品牌别名
        cls.get_brand_account_alias(init_resource_cls)

        # 获取 设计师 地区 内容占比
        cls.get_area_designer_content_ratio()

        # 初始化 内容id 对应 唯一 数字 用户 用户行为序列
        cls.initObjId2Index(init_resource_cls)

    # 检查 dssm query pb 是否已使用最新的
    @classmethod
    def checkQueryPbIsNewest(cls):
        query_pb_path_tmp = "/data/search_opt_model/dssm_recall/vec_recall_pb_tmp/bert_dssm_query"

        query_pb_path = "/data/search_opt_model/dssm_recall/vec_recall_pb/bert_dssm_query"

        faiss_model_path = "/data/search_opt_model/faiss_model4newrecall"

        # 判断文件
        if os.path.exists(query_pb_path_tmp) and os.path.exists(query_pb_path) and os.path.exists(faiss_model_path):
            # 临时pb文件
            query_pb_model_list_tmp, _, newest_query_pb_model_tmp = common4bert.get_models(query_pb_path_tmp)

            query_pb_time_tmp = -1
            if len(query_pb_model_list_tmp) > 0:
                query_pb_time_tmp = query_pb_model_list_tmp[0]

            # 判断 query pb 文件 和 faiss 文件是不是相同时间戳
            faiss_newest_time = -3
            faiss_model_list, _, _ = common4bert.get_models(faiss_model_path)
            if faiss_model_list != None and len(faiss_model_list) > 0:
                faiss_newest_model_file = faiss_model_list[0]
                faiss_newest_time = ((faiss_newest_model_file.split("_"))[1]).split(".")[0]

            if query_pb_time_tmp == faiss_newest_time:
                # 和最新的pb文件比较 是不是同一个文件
                query_pb_model_list, _, newest_query_pb_model = common4bert.get_models(query_pb_path)

                query_pb_time = -1
                if  len(query_pb_model_list) > 0:
                    query_pb_time = query_pb_model_list[0]

                if int(query_pb_time_tmp) > int(query_pb_time) and newest_query_pb_model_tmp != None and len(newest_query_pb_model_tmp) > 0 and\
                        Tool.get_dir_size(newest_query_pb_model_tmp) > 300:
                    # 文件大于300M
                    os.system("cp -R " + newest_query_pb_model_tmp + " " + query_pb_path)

    @classmethod
    def initBertService(cls):
        # 初始结巴服务
        cls.InitJieba()
        # 同义词功能
        cls.InitSynonym()

    @classmethod
    def initObjId2Index(cls, init_resource_cls):
        objId2IndexForUserProfile = ObjId2Index.getAllObjId2Index(init_resource_cls)
        cls.objId2IndexForUserProfile = objId2IndexForUserProfile

    @classmethod
    def get_top_query(cls, init_resource_cls):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 获取top query的数据
        '''
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        yesterday = int(str(yesterday).replace('-', ''))

        thirty_ago = datetime.date.today() - datetime.timedelta(days=30)
        thirty_ago = int(str(thirty_ago).replace('-', ''))
        sql_core = '''
            with b as (
            with a as (select
                    COALESCE(try(params['keyword']),try(params['tag']),try(params['k'])) as hot_key, 
                                                    count(*) as n
                                                    from oss.traffic.action_dwd a
                                                    where a.day  between %s and %s
                                                    and ( (a.type in ('search') and try(params['from']) in ('writeTag') ))
                                                    and coalesce(TRY(params['page']),TRY(params['p'])) = '1' -- 搜索加载一次（第一页）
                                                    group by
                                                    COALESCE(try(params['keyword']),try(params['tag']),try(params['k'])),uid
                                                    )
            select hot_key, count(*) as num from a group by hot_key order by num desc limit 2000
            )
            select * from b order by num asc''' % (thirty_ago, yesterday)

        top_query_infos = presto_execute(sql_core)
        log.logger.info("加载top query 成功")

        top_and_community_tag = set()

        for top_query_info in top_query_infos:
            keyword = top_query_info[0].strip()
            if len(keyword) > 0:
                top_and_community_tag.add(keyword)

        community_tags = Tag.get_all_community_tags(init_resource_cls)
        top_and_community_tag = top_and_community_tag | community_tags
        log.logger.info("加载社区标签完成")
        cls.community_tag = community_tags

        # log.logger.debug("top_and_community_tag:%s" %
        #                  str(top_and_community_tag))

        cls.top_and_community_tag = top_and_community_tag

    # @classmethod
    # def load_item_model(cls):
    #     itemModelPath = root_config["SearchServer"]["item_model"]
    #
    #     cls.item_model = fasttext.load_model(itemModelPath)

    @classmethod
    def get_area_designer_content_ratio(cls):
        '''
        @Author: xuzhongjie
        @param {*}
        @Description: 获取 地区设计师 内容占比
        '''
        hash_city_designer_ratio = {}

        with open(root_config["SearchServer"]["city_designer_ratio"], 'r', encoding='utf-8') as f:
            city_designer_ratios = f.read()
            city_designer_ratios_tmp = city_designer_ratios.split("\n")
            for city_designer_ratio_tmp in city_designer_ratios_tmp:
                if len(city_designer_ratio_tmp) > 0:
                    city_designer_ratio_list = city_designer_ratio_tmp.split(" ")

                    if len(city_designer_ratio_list) > 0:
                        hash_city_designer_ratio[str(city_designer_ratio_list[0])] = round(float(city_designer_ratio_list[1]) / 100, 2)

        cls.hash_city_designer_ratio = hash_city_designer_ratio

    @classmethod
    def load_item_model(cls):
        itemModelPath = root_config["SearchServer"]["item_model"]

        cls.item_model = fasttext.load_model(itemModelPath)

    @classmethod
    def get_brand_account_alias(cls, init_resource_cls):
        '''
            @Author: xuzhongjie
            @param {*}
            @Description: 获取 品牌 名称 英文名 别名
        '''
        all_brand_infos = Brand.getAllBrand(init_resource_cls)

        all_brand_name = set()
        tool_cls = Tool()

        if len(all_brand_infos) > 0:
            for brand_info in all_brand_infos:
                if "brand_name" in brand_info and len(brand_info["brand_name"]) > 0:
                    all_brand_name.add(tool_cls.T2S(brand_info["brand_name"]))

                if "en_brand_name" in brand_info and len(brand_info["en_brand_name"]) > 0:
                    all_brand_name.add(tool_cls.T2S(brand_info["en_brand_name"]))

                if "brand_alias" in brand_info and len(brand_info["brand_alias"]) > 0:
                    brand_alias_list = eval(brand_info["brand_alias"])

                    if len(brand_alias_list) > 0:
                        for brand_alias in brand_alias_list:
                            if len(brand_alias) > 0:
                                all_brand_name.add(tool_cls.T2S(brand_alias))

        if len(all_brand_name) > 0:
            cls.all_brand_name = all_brand_name

        print(cls.all_brand_name)

    @classmethod
    def InitAnnModel(cls):
        cls.SearchANN = SearchANN(root_config["faiss"]["search_use_gpu"])

    @classmethod
    def InitIntentClass(cls):
        cls.IntentModel = IntentClassModel(svm_model_dir, intent_pos_dir)

    @classmethod
    def init_term_weight_backstage(cls, init_resource_cls):
        SearchContent.add_all_query_term_weight_backstage(init_resource_cls)

    @classmethod
    def InitJieba(self):
        """
        jieba 动态线程启动
        Returns:

        """
        JiebaHHZ.loadUserDict()
        JiebaHHZ.del_jieba_default_words()


    @classmethod
    def InitSynonym(self):
        """
        同义词线程启动
        Returns:

        """
        Synonym.loadSynonym()

    @classmethod
    def start_dynamic_service(cls):
        cls.app.loop.create_task(JiebaThread.run(JiebaHHZ))
        cls.app.loop.create_task(SynonymThread.run(Synonym))


# if __name__ == "__main__":
#     InitService.get_top_query()
#     print(InitService.top_and_community_tag)