# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:08:29
Description: 获取点击数据（正负样本比例1:4)并且处理，用于训练rank模型

conda activate py38
nohup python -u click_data4recall_by_day.py > click_data4recall_by_day.out 2>&1 &
tail -f click_data4recall_by_day.out
ps -ef | grep click_data4recall_by_day.py
python click_data4recall_by_day.py
'''

import pickle
import os
import sys
import configparser
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
os.chdir(sys.path[0])
sys.path.append("../")
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from presto_con.topresto import python_presto
from common.common_log import Logger, get_log_name
from common.tool import Tool
from common.send_message import send_msg
from service.Tool.JiebaHHZ import JiebaHHZ
from common.get_ip import get_host_ip
from es.TotalArticle import TotalArticle
from es.Note import Note
from service.Init.InitResource import InitResource

    
init_resource_cls = InitResource()
init_resource_cls.initResource()

TotalArticleCls = TotalArticle(init_resource_cls)
noteEsCls = Note(init_resource_cls)

current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)

JiebaHHZ.loadUserDict()
JiebaHHZ.del_jieba_default_words()


class ClickData(object):
    '''
    Author: xiaoyichao
    param {*} self
    Description: 获取点击数据并且处理，用于训练rank模型
    '''

    def __init__(self):
        self.presto = python_presto()
        self.host = root_config["prosto"]['host']
        self.port = int(root_config["prosto"]['port'])
        self.user = root_config["prosto"]['user']
        self.password = root_config["prosto"]['password']

    def get_click_datas4recall(self, day):
        '''
        Author: xiaoyichao
        param {*}
        Description: 获取点击数据，目前需要增加对正负样本的比例控制
        直接用int 表示内容类型
            0 note
            1 整屋
            2 指南
            4 回答 note
            5 文章
        '''
        presto_con = self.presto.connect(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         password=self.password
                                         )
        log.logger.info("presto连接完成")
        sql = '''select
                        query as "搜索词",
                        obj_id as "内容id",
                        max(relevance) as "最大相关性得分",
                        max(is_valid) as "最大有效点击"
                    from oss.hhz.dws_search_order_model_v2
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null
                    group by day,query,obj_id,favorite_num,like_num,comment_num,score,add_day,wiki,user_identity_type,obj_type,designer_intention_group_tag
                    
                    ''' % (day)


        log.logger.info("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 完成")
        log.logger.info("开始解析数据")

        click_datas = {}
        corpus_obj_ids_set = set()
        i = 0

        for info in presto_cursor.fetchall():
            query = info[0]
            doc_id = info[1]
            relevance = info[2]
            is_valid = info[3]


            if query not in click_datas:
                click_datas[query] = {
                    'query': query, 'positive_ids_rele': [], 'negative_ids_rele': []}
                # pos_neg_num_dict[query] = {"pos_num": 0, "neg_num": 0}

            # 正样本
            if relevance > 0:
                if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                    relevance = 1
                    click_datas[query]['positive_ids_rele'].append(
                        (doc_id, relevance))
                    corpus_obj_ids_set.add(doc_id)
                else:
                    if relevance == 1 and is_valid == 1: # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                        click_datas[query]['positive_ids_rele'].append(
                            (doc_id, relevance))
                        corpus_obj_ids_set.add(doc_id)
                    else:
                        relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本
                        click_datas[query]['negative_ids_rele'].append(
                            (doc_id, relevance))
                        corpus_obj_ids_set.add(doc_id)

            # 负样本
            else:
                relevance = 0
                click_datas[query]['negative_ids_rele'].append((doc_id, relevance))
                corpus_obj_ids_set.add(doc_id)

            if i % 10000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i += 1

        log.logger.info("完成解析数据")
        log.logger.info("len(corpus_obj_ids_set)=%s"% len(corpus_obj_ids_set))
        
        return click_datas, corpus_obj_ids_set

    def get_corpus(self, obj_ids_list):
        '''
        Author: xiaoyichao
        param {*} self
        param {*} obj_ids_list
        Description: 获取obj_id 和 文本 组成的字典
        '''

        def cut(obj, basic_len4cut):
            '''
            Author: xiaoyichao
            param {*}
            Description: 按照长度且切分list,比如5000个数据切一次。但是有的训练数据已经被用户删除了，所以数据开会变少。
            '''
            return [obj[i:i+basic_len4cut] for i in range(0, len(obj), basic_len4cut)]

        corpus = {}
        # self.add_obj_pass_ids = []
        if len(obj_ids_list) > 0:
            basic_len4cut = 1000
            # objid_content_type_dict = self.get_objid_content_type_dict(obj_ids_list)
            content_type_ids_dict = Tool.get_content_type_ids_dict(obj_ids_list)
            # {content_type:1是文章，2是note，3是整屋，4是指南}
            for content_type, obj_ids in content_type_ids_dict.items():
                if content_type == 1 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessBlank(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 2 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut*10)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessNoteInfo(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 3 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessArticleInfo(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                elif content_type == 4 and len(obj_ids) > 0:
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessGuide(obj_ids_1D)
                        corpus.update(objid_title_remark_dict)
                else:
                    log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
                    err_log.logger.error(
                        "content_type_ids_dict错误的key:%s" % content_type)
        log.logger.info("corpus长度: "+str(len(corpus)))
        return corpus

    def get_samples(self, click_datas, corpus):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成samples
        '''
        import random
        from random import choice
        samples = []
        corpus_obj_ids_set = set(corpus.keys())
        query_set = set(click_datas.keys())

        for query, click_data in click_datas.items():
            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:

                # 曝光和点击数据的正负样本
                for i, pos_id_rele_data in enumerate(click_data['positive_ids_rele']):
                    if i < len(click_data['negative_ids_rele']):
                        neg_id_rele_data = click_data['negative_ids_rele'][i] # 负样本的list
                        pos_obj_id = pos_id_rele_data[0]
                        neg_obj_id = neg_id_rele_data[0]
                        # pos_rel = pos_id_rele_data[1]
                        # neg_rel = neg_id_rele_data[1]
                        
                        # if pos_obj_id in corpus_obj_ids_set and neg_obj_id in corpus_obj_ids_set:
                        if pos_obj_id in corpus_obj_ids_set:

                            pos_title = corpus[pos_obj_id]["title"]
                            pos_remark = corpus[pos_obj_id]["remark"]
                            
                            random_int = random.randint(1,100)
                            
                            if random_int < 40 :
                                if neg_obj_id in corpus_obj_ids_set:
                                    neg_title = corpus[neg_obj_id]["title"]
                                    neg_remark = corpus[neg_obj_id]["remark"]
                                    
                                    data4train = [query, pos_title, pos_remark, neg_title, neg_remark]

                                    samples.append(data4train)
                            
                            
                            elif random_int < 70 :
                            # 全局负采样的正负样本
                                other_query = random.sample(query_set, 1)[0]
                                global_negative_samplings = click_datas[other_query]['positive_ids_rele']
                                # if len(global_negative_samplings)==0:
                                #     global_negative_samplings = click_datas[other_query]['negative_ids_rele']
                                if other_query != query and len(global_negative_samplings) > 0:
                                    
                                    # 随机取一个其他query 下的正样本作为当前query下的负样本
                                    neg_id_rele_data = choice(global_negative_samplings)
                                    neg_obj_id = neg_id_rele_data[0]

                                    if neg_obj_id in corpus_obj_ids_set:
                                        neg_title = corpus[neg_obj_id]["title"]
                                        neg_remark = corpus[neg_obj_id]["remark"]
                                        
                                        data4train = [query, pos_title, pos_remark, neg_title, neg_remark]
                                        samples.append(data4train)
                                        log.logger.info("随机取一个其他query 下的正样本作为当前query下的负样本: %s"%neg_obj_id)
                                    else:
                                        log.logger.warning("%s  in corpus_obj_ids_set "%neg_obj_id)
                            
                            elif random_int < 95 :
                                #  获取一个note的不包含query的doc 作为负样本   
                                notes_es = noteEsCls.getRandomNegNoteForRecall(1, 1, "id, title, desc", query)
                                notes = notes_es["rows"]
                                if len(notes) > 0:
                                    for note in notes:
                                        neg_obj_id = note["id"]
                                        neg_title = note["title"]
                                        neg_remark = note["desc"][:64]
                                        data4train = [query, pos_title, pos_remark, neg_title, neg_remark]
                                        samples.append(data4train)
                                        log.logger.info("获取一个note的不包含query的doc 作为负样本: %s"%neg_obj_id)
                            else:
                                #  获取一个 整屋或者文章或者指南 的不包含query的doc 作为负样本   
                                total_article_es = TotalArticleCls.getRandomNegTotalArticleForRecall(1, 1, "id, title, fulltext", query)
                                total_articles = total_article_es["rows"]
                                if len(total_articles) > 0:
                                    for total_article in total_articles:
                                        neg_obj_id =  total_article["id"]
                                        neg_title = total_article["title"]
                                        neg_remark = total_article["fulltext"][:64]
                                        data4train = [query, pos_title, pos_remark, neg_title, neg_remark]
                                        samples.append(data4train)
                                        log.logger.info("获取一个 整屋或者文章或者指南 的不包含query的doc 作为负样本: %s"%neg_obj_id)


        log.logger.info("samples 长度: "+str(len(samples)))
        return samples

def write_samples_pkl(samples, output_file):
    '''
    Author: xiaoyichao
    param {*}
    Description: 将samples数据写入pkl文件
    '''
    output_file_path_list = output_file.split("/")
    output_file_parent_path = "/".join(output_file_path_list[:-1])
    if not os.path.exists(output_file_parent_path):
        os.makedirs(output_file_parent_path)
    with open(output_file, 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(samples, fo)
    log.logger.info("数据写入%s文件" % str(output_file))


def read_samples_pkl(pkl_file):
    '''
    Author: xiaoyichao
    param {*}
    Description: 从pkl文件中读取出samples数据
    '''
    with open(pkl_file, 'rb') as f:     # 读取pkl文件数据
        samples = pickle.load(f, encoding='bytes')
    return samples


def get_before_day(before_day):
    '''
    Author: xiaoyichao
    param {*}
    Description: 获取几天前的日期
    '''
    import datetime
    today = datetime.date.today()
    before_days = datetime.timedelta(days=before_day)
    before_day = today-before_days
    return str(before_day)


def merge_data(pkl_file_list):
    '''
    Author: xiaoyichao
    param {*}
    Description: 合并数据，但是要考虑是否要对数据去重复,或者保存字典，然后去重复
    '''
    samples_sum = []
    for pkl_file in pkl_file_list:
        samples = read_samples_pkl(pkl_file)
        samples_sum.extend(samples)


def get_click_data():
    # before_day = get_before_day(1)

    POS_NEG_RATION = 1  # 正负样本比例1:2
    corpus_obj_ids_set_sum = set()
    corpus_sum = {}
    clickdata = ClickData()
    begin_day = 1
    for day in range(begin_day, 5, 1):
        click_datas, corpus_obj_ids_set = clickdata.get_click_datas4recall(day=day)
        corpus_obj_ids_set_sum = corpus_obj_ids_set_sum | corpus_obj_ids_set
        corpus_obj_ids_set_add = corpus_obj_ids_set - corpus_obj_ids_set_sum
        log.logger.info("len(click_datas): " + str(len(click_datas)))
        if day == begin_day:
            corpus_obj_ids_set_add = corpus_obj_ids_set_sum
        corpus = clickdata.get_corpus(corpus_obj_ids_set_add)
        corpus_sum.update(corpus)
        log.logger.info("len(corpus_sum): " + str(len(corpus_sum)))
        samples = clickdata.get_samples(click_datas=click_datas, corpus=corpus_sum)
        log.logger.info("len(samples): " + str(len(samples)))
        pkl_file = os.path.join('/data/search_opt_model/recall/pkl_data/data_'+get_before_day(day)+'.pkl')
        write_samples_pkl(samples, pkl_file)
        send_msg("pkl文件数据4bert_recall生成完成:%s"%get_before_day(day))

        log.logger.info("数据读取完成 day = "+ str(day))
    log.logger.info("程序执行完毕")
    send_msg("pkl文件数据4bert_recall生成完成")


if __name__ == '__main__':
    get_click_data()
