# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:07:43
Description: 获取点击数据（正负样本比例1:4)并且处理，用于训练rank模型

conda activate py38
nohup python -u click_data4bert_by_day_1_4.py > click_data4bert_by_day_1_4.out 2>&1 &
tail -f click_data4bert_by_day_1_4.out
ps -ef | grep click_data4bert_by_day_1_4.py
'''
import os, sys, codecs, json
os.environ['TF_KERAS'] = '1'
import pickle

import configparser
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
os.chdir(sys.path[0])
sys.path.append("../")
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from presto_con.topresto import python_presto
from common.common_log import Logger, get_log_name
from common.tool import Tool
import random
import tensorflow as tf
from service.Tool.JiebaHHZ import JiebaHHZ
from db.hhzSearch.ObjId2Index import ObjId2Index
import numpy as np


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


def get_tf(query, sentence, have_jiebad=False, max_len=128):
    '''
    Author: xiaoyichao
    param {*}
    Description: 获取query 在doc中的词频
    '''
    tf_num = 0
    query_word_list = JiebaHHZ.SplitWords(query)
    if have_jiebad:
        sentence_word_list = sentence[:max_len]  # max_len个词
        sentence = "".join(sentence)
        sentence = sentence[:max_len]  # max_len个字
    else:
        sentence_word_list = JiebaHHZ.SplitWords(sentence[:max_len], isGetUnique = False)
    for query_word in query_word_list:
        for sentence_word in sentence_word_list:
            if sentence_word == query_word:
                tf_num += 1
    if query in sentence:  # 给 绿色-沙发 的 数据加分，防止“绿色的植物，沙发也很软”这样query的两个词分家的数据得到高分。
        tf_num += 1
    return tf_num


def standard_scaler(data):
    '''
    Author: xiaoyichao
    param {*}
    Description: 对数值型数据进行标准化，使其数据分数的均值为0附近 ，方差为1
    '''    
    mm = StandardScaler()
    mm_data = mm.fit_transform(data)
    return mm_data


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

    def get_click_datas4bert(self, day):
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
                        is_valid as "是否 是有效点击",
                        relevance as "相关性得分",
                        uid as "用户uid",
                        search_request_id as "search id"
                    from oss.hhz.dws_search_user_personalization_daily_report
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null 
                    and is_valid is not null and uid is not null and search_request_id is not null
                    ''' % (day)


        log.logger.info("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("presto 执行 完成")
        log.logger.info("开始解析数据")

        click_datas = {}
        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        neg_objIds = set()

        i = 0

        for info in presto_cursor.fetchall():
            query = info[0]
            doc_id = info[1]
            doc_id = doc_id.strip()
            if len(doc_id) == 0:
                continue

            is_valid = info[2]
            relevance = info[3]
            uid = info[4]
            search_request_id = info[5]

            unique_str = str(query) + "_" + str(uid) + "_" + str(search_request_id)
            if unique_str not in click_datas:
                click_datas[unique_str] = {
                    'query': query, 'positive_ids_rele': set(), 'negative_ids_rele': set()}

            one_sample = (query, doc_id, is_valid, relevance)

            # 正样本
            if relevance > 0:
                if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                    click_datas[unique_str]['positive_ids_rele'].add(one_sample)
                    corpus_obj_ids_set.add(doc_id)
                else:
                    # 正负样本比例对
                    if relevance == 1 and is_valid == 1: # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                        click_datas[unique_str]['positive_ids_rele'].add(one_sample)
                        corpus_obj_ids_set.add(doc_id)
                    else:
                        relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本

                        one_sample = list(one_sample)
                        one_sample[3] = relevance
                        one_sample = tuple(one_sample)

                        click_datas[unique_str]['negative_ids_rele'].add(one_sample)
                        corpus_obj_ids_set.add(doc_id)
                        neg_objIds.add(doc_id)

            # 负样本
            else:
                click_datas[unique_str]['negative_ids_rele'].add(one_sample)
                corpus_obj_ids_set.add(doc_id)
                neg_objIds.add(doc_id)

            if i % 10000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i += 1

        log.logger.info("完成解析数据")
        return click_datas, corpus_obj_ids_set, neg_objIds

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
            basic_len4cut = 500
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
                    obj_ids_2D = cut(obj_ids, 5000)
                    for obj_ids_1D in obj_ids_2D:
                        _, _, _, _, objid_title_remark_dict = getProcessNoteInfo(obj_ids_1D, use_presto=True)
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

    def get_samples(self, click_datas, corpus, neg_objIds, max_relevance=3, pos_neg_ration=4, max_len=32):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成samples
        '''
        samples = []
        corpus_obj_ids_set = set(corpus.keys())

        neg_objIds_tmp = list(neg_objIds & corpus_obj_ids_set)

        positive_data_num = 0
        for unique_str, click_data in click_datas.items():
            query = click_data['query']

            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:
                positive_ids_set = set(click_data['positive_ids_rele'])

                for pos_id_rele_data in click_data['positive_ids_rele']:
                    #  (query, doc_id, is_valid, relevance)
                    posObjId = pos_id_rele_data[1]

                    if posObjId in corpus_obj_ids_set:
                        posTitle = corpus[posObjId]["title"]
                        posRemark = corpus[posObjId]["remark"]

                        # 负样本
                        negative_ids_rele_list = list(click_data['negative_ids_rele'])
                        random.shuffle(negative_ids_rele_list)

                        neg_index = 0
                        for neg_id_rele_data in negative_ids_rele_list:
                            negObjId = neg_id_rele_data[1]
                            neg_index = neg_index + 1

                            # 控制比例 只取一条
                            if neg_index > 1:
                                break

                            if negObjId in corpus_obj_ids_set:
                                negTitle = corpus[negObjId]["title"]
                                negRemark = corpus[negObjId]["remark"]
                                data4train = [query, posTitle, posRemark, negTitle, negRemark]

                                samples.append(data4train)

                        # 全局负采样
                        neg_objIds_tmp_sample = random.sample(neg_objIds_tmp, 20)
                        neg_index = 0

                        for objId in neg_objIds_tmp_sample:
                            if objId in positive_ids_set:
                                continue

                            neg_index = neg_index + 1

                            if neg_index > pos_neg_ration:
                                break

                            negTitle = corpus[objId]["title"]
                            negRemark = corpus[objId]["remark"]

                            data4train = [query, posTitle, posRemark, negTitle, negRemark]

                            samples.append(data4train)

        log.logger.info("samples 长度: "+str(len(samples)))
        log.logger.info("所有正样本的 数量: "+str(positive_data_num))
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

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(qt, qs, posDt, posDs, negDt, negDs):
    """ [query, posTitle, posRemark, negTitle, negRemark]
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'qt' : _bytes_feature(qt),
        'qs': _bytes_feature(qs),
        'posDt': _bytes_feature(posDt),
        'posDs': _bytes_feature(posDs),
        'negDt': _bytes_feature(negDt),
        'negDs': _bytes_feature(negDs),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs

token_dict = {}
with codecs.open("/home/adm_rsync_dir/search_op/vocab.txt", "r", "utf-8") as fr:
    for line in fr:
        line = line.strip()
        token_dict[line] = len(token_dict)
tokenizerGlobal = Tokenizer(token_dict)

def write_samples_tfrecord(samples, output_file):
    '''
    Author: xiaoyichao
    param {*}
    Description: 将samples数据写入pkl文件
    '''
    output_file_path_list = output_file.split("/")
    output_file_parent_path = "/".join(output_file_path_list[:-1])
    if not os.path.exists(output_file_parent_path):
        os.makedirs(output_file_parent_path)

    writer = tf.io.TFRecordWriter(output_file)

    def parse_data(line, max_len):
        query = line[0]
        posTitle = line[1]
        posRemark = line[2]
        negTitle = line[3]
        negRemark = line[4]

        # qt, qs = tokenizer.encode(first = line[0], second = , max_len=max_len)
        qt, qs = tokenizerGlobal.encode(query, max_len=max_len)

        posDt, posDs = tokenizerGlobal.encode(first=posTitle, second=posRemark, max_len=max_len)

        negDt, negDs = tokenizerGlobal.encode(first=negTitle, second=negRemark, max_len=max_len)

        return qt, qs, posDt, posDs, negDt, negDs

    for sample in samples:
        qt, qs, posDt, posDs, negDt, negDs = parse_data(sample, 64)
        writer.write(serialize_example(tf.io.serialize_tensor(qt), tf.io.serialize_tensor(qs), tf.io.serialize_tensor(posDt), tf.io.serialize_tensor(posDs), tf.io.serialize_tensor(negDt), tf.io.serialize_tensor(negDs)))

    writer.flush()
    writer.close()

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

from service.Init.InitResource import InitResource
# 获取 内容 id 对应 index
def getObjId2Index():
    InitResourceCls = InitResource()
    InitResourceCls.initDbSearchConnect()

    objId2Index = ObjId2Index.getAllObjId2Index(InitResourceCls)
    InitResourceCls.closeDbSearchConnect()

    return objId2Index

def get_click_data():
    # before_day = get_before_day(1)
    # 获取 内容 id 对应 数字

    POS_NEG_RATION = 4  # 正负样本比例1:4
    max_len = 64
    max_relevance = 3
    corpus_obj_ids_set_sum = set()
    corpus_sum = {}

    clickdata = ClickData()
    for day in range(2, 60, 1):
        click_datas, corpus_obj_ids_set, neg_objIds = clickdata.get_click_datas4bert(day=day)

        corpus_obj_ids_set_add = corpus_obj_ids_set - corpus_obj_ids_set_sum

        log.logger.info("len(click_datas): " + str(len(click_datas)))

        corpus = clickdata.get_corpus(corpus_obj_ids_set_add)

        corpus_sum.update(corpus)

        corpus_obj_ids_set_sum = corpus_obj_ids_set_sum | corpus_obj_ids_set
        log.logger.info("len(corpus_sum): " + str(len(corpus_sum)))
        samples = clickdata.get_samples(click_datas=click_datas, corpus=corpus_sum, max_relevance=max_relevance, pos_neg_ration=POS_NEG_RATION, max_len=max_len, neg_objIds = neg_objIds)
        log.logger.info("len(samples): " + str(len(samples)))

        pkl_file = os.path.join('/data/search_opt_model/dssm_recall/rank_data_vec_recall/data_'+get_before_day(day)+'.pkl')

        write_samples_pkl(samples, pkl_file)

        log.logger.info("数据读取完成"+ str(day))
    log.logger.info("程序执行完毕")
    # send_msg("pkl文件数据 rank_data_more_1_4_tf 生成完成")

if __name__ == '__main__':
    get_click_data()
