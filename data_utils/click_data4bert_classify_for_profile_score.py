# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2022-08-02 16:09:49
Description: 获取点击数据（正负样本比例1:4)并且处理，用于训练rank模型

conda activate py38
nohup python -u click_data4bert_by_day_1_4.py > click_data4bert_by_day_1_4.out 2>&1 &
tail -f click_data4bert_by_day_1_4.out
ps -ef | grep click_data4bert_by_day_1_4.py
'''

import pickle
import os
import sys
import configparser
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
os.chdir(sys.path[0])
sys.path.append("../")
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from presto_con.topresto import python_presto
from common.common_log import Logger, get_log_name
from common.tool import Tool
from common.tool import Tool
from common.send_message import send_msg
from common.get_ip import get_host_ip
from service.Tool.JiebaHHZ import JiebaHHZ


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


folder_name =  "search_opt_model"


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

        # 搜索词 的 搜索次数
        sql = '''
        select query, count(*) as query_num from  ( select
                        query,
                        search_request_id,
                        max(is_valid) as max_is_valid,
                         max(relevance) as max_relevance
                    from hive.hhz.dws_search_user_personalization_daily_report
                    where  day = #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null and search_request_id is not null
                    group by query, search_request_id ) where (max_is_valid = 1 and max_relevance = 1) OR (max_relevance > 1)  group by query  order by query_num desc''' % (day)

        log.logger.info("搜索词 的 搜索次数 presto cursor sql: " + sql)
        presto_cursor = presto_con.cursor()
        log.logger.info("搜索词 的 搜索次数 presto cursor 完成")
        presto_cursor.execute(sql)
        log.logger.info("搜索词 的 搜索次数 执行 完成")
        log.logger.info("开始解析数据")

        hash_query_num = {}

        for info in presto_cursor.fetchall():
            query = info[0]
            query_num = info[1]

            hash_query_num[query] = query_num

        presto_cursor.close()


        sql = '''select
                        query as "搜索词",
                        obj_id as "内容id",
                        max(relevance) as "最大相关性得分",
                        favorite_num as "内容的总收藏数",
                        like_num as "内容的总点赞数",
                        comment_num as "内容的总评论数",
                        score as "内容的后台质量分数",
                        wiki as "是否挂了wiki",
                        user_identity_type as "发布内容的用户身份",
                        obj_type as "内容类型",
                        add_day as "内容的首次发布时间",
                        day as "搜索行为发生的日期",
                        max(is_valid) as "最大有效点击"
                    from hive.hhz.dws_search_order_model_v2
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
        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        obj_type_dict_keys = set(obj_type_dict.keys())
        user_identity_type_list = ["设计师用户", "普通用户", "个人V认证用户", "品牌用户", "机构V认证用户", "装修公司用户", "未认证设计师用户", "未知用户类型"]
        obj_type_list = ["note", "整屋", "指南", "回答的note", "文章", "未知内容类型"]
        wiki_dict = {"是": 1, "否": 0}

        i = 0

        for info in presto_cursor.fetchall():
            query = info[0]
            doc_id = info[1]

            if len(doc_id) < 10 :
                continue

            relevance = info[2]
            favorite_num = info[3]
            like_num = info[4]
            comment_num = info[5]
            score = info[6]
            wiki = info[7]
            user_identity_type = info[8]
            obj_type_num = Tool.getTypeByObjId(doc_id)
            add_day = info[10]
            day = info[11]
            is_valid = info[12]
            # day = time.strftime("%Y-%m-%d", time.localtime()).replace("-", "")
            # designer_intention_group_tag = info[13]

            '''数据预处理'''
            favorite_num = 0 if favorite_num is None else favorite_num
            like_num = 0 if like_num is None else like_num
            comment_num = 0 if comment_num is None else comment_num

            wiki = "否" if wiki is None else wiki
            user_identity_type = "未知用户类型" if user_identity_type is None else user_identity_type
            # obj_type = "未知内容类型" if obj_type_num is None else obj_type_dict[obj_type_num]
            obj_type = "未知内容类型" if obj_type_num not in obj_type_dict_keys else obj_type_dict[obj_type_num]

            if obj_type == "note" or obj_type == "回答的note":
                if score is None:
                    score = 0
                else:
                    if score < 60:
                        score = 60
            score = 0 if score is None else score

            if add_day and day:
                add_day = str(add_day)
                day = str(day)
                add_day = datetime(year=int(add_day[0:4]), month=int(
                    add_day[4:6]), day=int(add_day[6:8]))
                day = datetime(year=int(day[0:4]), month=int(
                    day[4:6]), day=int(day[6:8]))
                interval_days = (day - add_day).days
            else:
                interval_days = 0

            if query not in click_datas:
                click_datas[query] = {
                    'query': query, 'positive_ids_rele': set(), 'negative_ids_rele': set()}
                # pos_neg_num_dict[query] = {"pos_num": 0, "neg_num": 0}

            # 正样本
            if relevance > 0:
                if relevance > 1:  # 有交互行为的数据是相关性>1 的正样本
                    click_datas[query]['positive_ids_rele'].add(
                        (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                    corpus_obj_ids_set.add(doc_id)
                else:
                    # if relevance == 1: # 正负样本比例不对
                    #     relevance = is_valid  # 正负样本比例不对
                    # 正负样本比例对
                    if relevance == 1 and is_valid == 1: # relevance == 1 是发生了点击行为的，is_valid == 1是有效点击，两个条件都满足的才是真正的相关性为1
                        click_datas[query]['positive_ids_rele'].add(
                            (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                        corpus_obj_ids_set.add(doc_id)
                    else:
                        relevance = 0  # 点击了但是是无效点击是负样本 或者根本就没与欧点击的样本也是负样本
                        click_datas[query]['negative_ids_rele'].add(
                            (doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                        corpus_obj_ids_set.add(doc_id)

            # 负样本
            else:
                click_datas[query]['negative_ids_rele'].add((doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                corpus_obj_ids_set.add(doc_id)

            if i % 10000 == 0:
                log.logger.info("目前解析的数据量:"+str(i))
            i += 1

        log.logger.info("完成解析数据")
        return click_datas, corpus_obj_ids_set, hash_query_num

    def get_corpus(self, obj_ids_list):
        '''
        Author: xiaoyichao
        param {*} self
        param {*} obj_ids_list
        Description: 获取obj_id 和 文本 组成的字典
        '''
        from data_utils.Thread.dataProcessFromDb import dataProcessFromDb
        from data_utils.Thread.dataProcessInsert import dataProcessInsert
        from multiprocessing import Queue
        import multiprocessing

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
                thread = dataProcessFromDb(thread_id, objIdQueue, data_queue, objNum, content_type, objType,
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

    def get_samples(self, click_datas, corpus, max_relevance=3, pos_neg_ration=4, max_len=32):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成samples
        '''
        samples = []

        eval_samples_classify = {}

        corpus_obj_ids_set = set(corpus.keys())

        for query, click_data in click_datas.items():
            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:
                # 正样本 为 空的情况下 没必要训练该搜索词
                pos_cnt_tmp = 0
                # 获取 正样本 数量
                for pos_id_rele_data in click_data['positive_ids_rele']:
                    objId = pos_id_rele_data[0]

                    if objId in corpus_obj_ids_set:
                        # 这个case 会把那些同义词的结果都变成负样本
                        title = corpus[objId]["title"]
                        remark = corpus[objId]["remark"]

                        if query in title[:max_len] or query in remark[:max_len]:
                            pos_cnt_tmp += 1

                # 正样本为空 没必要训练
                if pos_cnt_tmp == 0:
                    continue

                if query not in eval_samples_classify:
                    eval_samples_classify[query] = []

                neg_cnt = 0
                pos_cnt = 0
                for pos_id_rele_data in click_data['positive_ids_rele']:
                    if pos_id_rele_data[0] in corpus_obj_ids_set:
                        # pos_cnt += 1
                        # label = float(pos_id_rele_data[1]/max_relevance)
                        # doc = corpus[pos_id_rele_data[0]]
                        # samples.append(InputExample(
                        #     texts=[query, doc], label=label))

                        # 这个case 会把那些同义词的结果都变成负样本
                        title = corpus[pos_id_rele_data[0]]["title"]
                        remark = corpus[pos_id_rele_data[0]]["remark"]
                        title_tf_num = get_tf(query, title, have_jiebad=False, max_len=128)  # 这个max_len不需要跟bert的max_len一样
                        remark_tf_num = get_tf(query, remark, have_jiebad=False, max_len=128)
                        if query in title[:max_len] or query in remark[:max_len]:
                            pos_cnt += 1
                            label = str(pos_id_rele_data[1])
                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                            for d in pos_id_rele_data[2:]:
                                data4train.append(d)
                            samples.append(data4train)

                            eval_samples_classify[query].append(data4train)
                            # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                        else:
                            # if neg_cnt < pos_cnt*pos_neg_ration:
                            neg_cnt += 1
                            label = 0
                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                            for d in pos_id_rele_data[2:]:
                                data4train.append(d)
                            samples.append(data4train)

                            eval_samples_classify[query].append(data4train)
                            # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                            # else:
                            #     break


                for neg_id_rele_data in click_data['negative_ids_rele']:
                    if neg_id_rele_data[0] in corpus_obj_ids_set:
                        if neg_cnt < pos_cnt*pos_neg_ration:
                            neg_cnt += 1
                            # label = float(neg_id_rele[1]/max_relevance)
                            label = 0
                            title = corpus[neg_id_rele_data[0]]["title"]
                            remark = corpus[neg_id_rele_data[0]]["remark"]
                            title_tf_num = get_tf(query, title, have_jiebad=False, max_len=128)  # 这个max_len不需要跟bert的max_len一样
                            remark_tf_num = get_tf(query, remark, have_jiebad=False, max_len=128)

                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type
                            for d in neg_id_rele_data[2:]:
                                data4train.append(d)
                            samples.append(data4train)

                            eval_samples_classify[query].append(data4train)
                        else:
                            break

        log.logger.info("samples 长度: "+str(len(samples)))
        return samples, eval_samples_classify

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

    POS_NEG_RATION = 4  # 正负样本比例1:4
    max_len = 64
    max_relevance = 3
    corpus_obj_ids_set_sum = set()
    corpus_sum = {}
    clickdata = ClickData()
    for day in range(1, 2, 1):
        click_datas, corpus_obj_ids_set, hash_query_num = clickdata.get_click_datas4bert(day=day)

        corpus_obj_ids_set_add = corpus_obj_ids_set - corpus_obj_ids_set_sum

        log.logger.info("len(click_datas): " + str(len(click_datas)))

        corpus = clickdata.get_corpus(corpus_obj_ids_set_add)

        corpus_sum.update(corpus)

        corpus_obj_ids_set_sum = corpus_obj_ids_set_sum | corpus_obj_ids_set

        log.logger.info("len(corpus_sum): " + str(len(corpus_sum)))

        samples, eval_classify_samples = clickdata.get_samples(click_datas=click_datas, corpus=corpus_sum, max_relevance=max_relevance, pos_neg_ration=POS_NEG_RATION, max_len=max_len)
        log.logger.info("len(samples): " + str(len(samples)))

        npy_classify_file = os.path.join('/data', folder_name, 'topk_opt/rank_data_classify_for_profile_score/eval_classify_'+get_before_day(day)+'.npy')

        hash_eval_classify_data = {
            "hash_query_num" : hash_query_num,
            "eval_classify_samples" : eval_classify_samples
        }

        np.save(npy_classify_file, hash_eval_classify_data)

        log.logger.info("数据读取完成"+ str(day))
    log.logger.info("程序执行完毕")
    # send_msg("pkl文件数据 rank_data_classify 生成完成")


if __name__ == '__main__':
    get_click_data()
