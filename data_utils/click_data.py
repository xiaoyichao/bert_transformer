# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-10-12 22:51:42
LastEditTime: 2021-12-01 14:58:14
Description: 获取点击数据并且处理，用于训练rank模型
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
        sentence_word_list = JiebaHHZ.SplitWords(sentence[:max_len])
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

    def get_click_datas(self, before_day, end_day, top_k=5, use_use_way=None, before_day_is_day=True, use_repeat_data=False):
        '''
        Author: xiaoyichao
        param {use_day:是否按照天展示数据，这样数据会更多，pos_neg_ration:正负样本比例, top_k:取第几个数据, before_day:开始时间（可以是含义是几天前（int）或者某个具体日期(str)）, end_day：结束时间，含义是几天前}
        Description: 曝光后点击
        '''

        presto_con = self.presto.connect(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         password=self.password
                                         )
        print("presto连接完成")

        if use_use_way == "max":
            if before_day_is_day:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                max(relevance) as "平均相关性得分",
                            from oss.hhz.dws_search_order_model_v2
                            where  day between %d and  #{date-%s,yyyyMMdd}
                            group by query,obj_id ''' % (before_day, end_day)

            else:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                max(relevance) as "平均相关性得分"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                            group by query,obj_id ''' % (before_day, end_day)

        if use_use_way == "avg":
            if before_day_is_day:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                avg(relevance) as "平均相关性得分"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between %d and  #{date-%s,yyyyMMdd}
                            group by query,obj_id ''' % (before_day, end_day)

            else:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                avg(relevance) as "平均相关性得分"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                            group by query,obj_id ''' % (before_day, end_day)
        if use_use_way == "top_k":  # 使用tok K 的数据
            if before_day_is_day:
                sql = '''
                        select query as "搜索词",
                                obj_id as "内容id",
                                relevance "最大相关性得分"
                        from
                            (select
                                query,
                                obj_id,
                                relevance,
                                row_number() over(partition by query,obj_id order by relevance desc) as aa
                            from oss.hhz.dws_search_order_model_v2
                            where  day between %d and  #{date-%s,yyyyMMdd}
                            )
                        where  aa = %s

                        union

                        select t1.query as "搜索词",
                            t1.obj_id as "内容id",
                            t1."最大相关性得分"
                        from
                            (select
                                query,
                                obj_id,
                                max(relevance) as "最大相关性得分"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between %d and  #{date-%s,yyyyMMdd}
                            group by query,obj_id
                            )t1
                        inner join (
                            select query,obj_id
                            from
                                (select
                                    query ,
                                    obj_id ,
                                    row_number() over(partition by query,obj_id order by relevance desc) as aa
                                from oss.hhz.dws_search_order_model_v2
                                where  day between %d and  #{date-%s,yyyyMMdd}
                                )
                            group by 1,2
                            having max(aa) < %s
                            )t2 on t1.query = t2.query and t1.obj_id = t2.obj_id
                            ''' % (before_day, end_day, top_k, before_day, end_day, before_day, end_day, top_k)

            else:
                sql = '''
                        select query as "搜索词",
                                obj_id as "内容id",
                                relevance "最大相关性得分"
                        from
                            (select
                                query,
                                obj_id,
                                relevance,
                                row_number() over(partition by query,obj_id order by relevance desc) as aa
                            from oss.hhz.dws_search_order_model_v2
                            where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                            )
                        where  aa = %s

                        union

                        select t1.query as "搜索词",
                            t1.obj_id as "内容id",
                            t1."最大相关性得分"
                        from
                            (select
                                query,
                                obj_id,
                                max(relevance) as "最大相关性得分"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                            group by query,obj_id
                            )t1
                        inner join (
                            select query,obj_id
                            from
                                (select
                                    query ,
                                    obj_id ,
                                    row_number() over(partition by query,obj_id order by relevance desc) as aa
                                from oss.hhz.dws_search_order_model_v2
                                where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                                )
                            group by 1,2
                            having max(aa) < %s
                            )t2 on t1.query = t2.query and t1.obj_id = t2.obj_id
                            ''' % (before_day, end_day, top_k, before_day, end_day, before_day, end_day, top_k)

        if use_repeat_data:  # 按照天展示数据，这个是个不完整的实验。因为click_datas[query] = {'query': query, 'positive_ids_rele': set(), 'negative_ids_rele': set()}。 set()需要换成list
            if before_day_is_day:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                relevance as "相关性得分",
                                uid as "搜索者uid",
                                day as "搜索行为发生的日期"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between %d and  #{date-%s,yyyyMMdd}
                            group by day, query, obj_id, uid, relevance  ''' % (before_day, end_day)

            else:
                sql = ''' select
                                query as "搜索词",
                                obj_id as "内容id",
                                relevance as "相关性得分",
                                uid as "搜索者uid",
                                day as "搜索行为发生的日期"
                            from oss.hhz.dws_search_order_model_v2
                            where  day between #{date-%s,yyyyMMdd} and  #{date-%s,yyyyMMdd}
                            group by day, query, obj_id, uid, relevance  ''' % (before_day, end_day)

        print("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        print("presto cursor 完成")
        presto_cursor.execute(sql)
        print("presto 执行 完成")

        print("开始解析数据到click_datas = {}")
        click_datas = {}
        corpus_obj_ids_set = set()
        pos_neg_num_dict = {}
        if use_use_way == "max":
            relevance_threshold = 2
        else:
            relevance_threshold = 1

        for info in presto_cursor.fetchall():
            query = info[0]
            doc_id = info[1]
            relevance = info[2]
            pos_num = 0
            neg_num = 0

            if query not in click_datas and query is not None:
                click_datas[query] = {
                    'query': query, 'positive_ids_rele': set(), 'negative_ids_rele': set()}
                pos_neg_num_dict[query] = {"pos_num": 0, "neg_num": 0}

            if query in click_datas:

                if relevance >= relevance_threshold:
                    click_datas[query]['positive_ids_rele'].add((doc_id, relevance))
                    # click_datas[query]['positive_ids_rele'].add(doc_id)
                    corpus_obj_ids_set.add(doc_id)
                    pos_num = pos_neg_num_dict[query]["pos_num"]
                    pos_neg_num_dict[query]["pos_num"] = pos_num+1
                else:
                    # if pos_neg_num_dict[query]["neg_num"] < (pos_neg_ration+1)*pos_neg_num_dict[query]["pos_num"]: # pos_neg_ration+1是为了防止下一步的获取doc str 的时候，有些数据被用户删除，导致比例不够
                    relevance = 0
                    corpus_obj_ids_set.add(doc_id)
                    click_datas[query]['negative_ids_rele'].add((doc_id, relevance))
                    neg_num = pos_neg_num_dict[query]["neg_num"]
                    pos_neg_num_dict[query]["neg_num"] = neg_num+1

        presto_cursor.close()
        presto_con.close()
        print("query数量(click_datas数量): "+str(len(click_datas)))
        print("corpus_obj_ids_set数量: "+str(len(corpus_obj_ids_set)))
        print("日志中正样本数量: "+str(pos_num))
        print("日志中负样本数量: "+str(neg_num))

        return click_datas, corpus_obj_ids_set

    def get_click_datas4bert(self, before_day, end_day):
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
        print("presto连接完成")
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
                        day as "搜索行为发生的日期"
                    from oss.hhz.dws_search_order_model_v2
                    where  day between #{date-%s,yyyyMMdd} and #{date-%s,yyyyMMdd} and query is not null and obj_id is not null and relevance is not null 
                    group by day,query,obj_id,favorite_num,like_num,comment_num,score,add_day,wiki,user_identity_type,obj_type,designer_intention_group_tag
                    ''' % (before_day, end_day)

        print("presto_cursor: "+sql)
        presto_cursor = presto_con.cursor()
        print("presto cursor 完成")
        presto_cursor.execute(sql)
        print("presto 执行 完成")
        print("开始解析数据")

        click_datas = {}
        # pos_neg_num_dict = {}
        corpus_obj_ids_set = set()

        user_identity_type_list = ["设计师用户", "普通用户", "个人V认证用户", "品牌用户", "机构V认证用户", "装修公司用户", "未认证设计师用户", "未知用户类型"]
        obj_type_dict = {0: "note", 1: "整屋", 2: "指南", 4: "回答的note", 5: "文章"}
        obj_type_list = ["note", "整屋", "指南", "回答的note", "文章", "未知内容类型"]

        # obj_type_dict = dict(zip(obj_type_dict.values(), obj_type_dict.keys()))
        wiki_dict = {"是": 1, "否": 0}

        i = 0

        for info in presto_cursor.fetchall():
            if i == 0:
                print("拉取数据完成，数据量:", len(presto_cursor.description))
            query = info[0]
            doc_id = info[1]
            relevance = info[2]
            favorite_num = info[3]
            like_num = info[4]
            comment_num = info[5]
            score = info[6]
            wiki = info[7]
            user_identity_type = info[8]
            # obj_type = info[9] # obj_type_dict = {"文章": 1, "单图": 2, "多图": 3, "整屋": 4, "视频": 5, "纯文字": 6, "未知":0}  这个目前没使用
            obj_type_num = Tool.getTypeByObjId(doc_id)
            add_day = info[10]
            day = info[11]
            # day = time.strftime("%Y-%m-%d", time.localtime()).replace("-", "")
            # designer_intention_group_tag = info[12]

            '''数据预处理'''
            favorite_num = 0 if favorite_num is None else favorite_num
            like_num = 0 if like_num is None else like_num
            comment_num = 0 if comment_num is None else comment_num

            wiki = "否" if wiki is None else wiki
            user_identity_type = "未知用户类型" if user_identity_type is None else user_identity_type
            obj_type = "未知内容类型" if obj_type_num is None else obj_type_dict[obj_type_num]

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
                click_datas[query]['positive_ids_rele'].add((doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                corpus_obj_ids_set.add(doc_id)

            # 负样本
            if relevance == 0:
                click_datas[query]['negative_ids_rele'].add((doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type))
                corpus_obj_ids_set.add(doc_id)

            if i % 10000 == 0:
                print("解析的数据量:", str(i))
            i += 1

        print("完成解析数据")
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
                    obj_ids_2D = cut(obj_ids, basic_len4cut)
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
        print("corpus长度: "+str(len(corpus)))
        return corpus

    def get_samples(self, click_datas, corpus, max_relevance=3, pos_neg_ration=4, max_len=32):
        '''
        Author: xiaoyichao
        param {*}
        Description: 将doc_id转化为文本，生成train_samples
        '''
        samples = []
        corpus_obj_ids_set = set(corpus.keys())

        for query, click_data in click_datas.items():
            if len(click_data['positive_ids_rele']) > 0 and len(click_data['negative_ids_rele']) > 0:
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
                            label = round(float(pos_id_rele_data[1]/max_relevance), 2)
                            data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                            # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                            for d in pos_id_rele_data[2:]:
                                data4train.append(d)
                            samples.append(data4train)
                            # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                        else:
                            if neg_cnt < pos_cnt*pos_neg_ration:
                                neg_cnt += 1
                                label = 0
                                data4train = [query, title, remark, str(label), title_tf_num, remark_tf_num]
                                # pos_id_rele_data = [doc_id, relevance, favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]
                                for d in pos_id_rele_data[2:]:
                                    data4train.append(d)
                                samples.append(data4train)
                                # samples = [doc_id, relevance, title_tf_num, remark_tf_num , favorite_num, like_num, comment_num, score, interval_days, wiki, obj_type]

                            else:
                                break

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
                        else:
                            break
        print("samples 长度: "+str(len(samples)))
        return samples

    def write_samples_pkl(self, samples, output_file):
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
        print("数据写入%s文件" % str(output_file))

    def read_samples_pkl(self, pkl_file):
        '''
        Author: xiaoyichao
        param {*}
        Description: 从pkl文件中读取出samples数据
        '''
        with open(pkl_file, 'rb') as f:     # 读取pkl文件数据
            samples = pickle.load(f, encoding='bytes')
        return samples


if __name__ == '__main__':
    # samples = [["query", "title", "remark", "1"] for i in range(100)]
    # clickData = ClickData()
    # # clickData.write_samples_pkl(
    # #     samples, "/data/xiaoyichao/projects/search_opt/data_utils/train.pkl")
    # samples = clickData.read_samples_pkl("/data/xiaoyichao/projects/search_opt/rank/../data/pkl4bert/train.pkl")
    # print(len(samples))
    # obj_ids_1D = ["0004h3a01001lfc8"]
    # _, _, _, _, objid_title_remark_dict = getProcessArticleInfo(obj_ids_1D)
    # print(objid_title_remark_dict)
    POS_NEG_RATION = 4  # 正负样本比例1:4
    max_len = 64
    max_relevance = 3
    NUM_MAX_DEV_NEGATIVES = 1e5
    use_use_way = "max"
    from common.get_ip import get_host_ip
    if get_host_ip() == "172.16.10.10":
        NUM_DEV_QUERIES = 300
    else:
        NUM_DEV_QUERIES = 1e5
    clickdata = ClickData()
    train_click_datas, train_corpus_obj_ids_set = clickdata.get_click_datas4bert(before_day=2, end_day=2)
    print("len(train_click_datas): " + str(len(train_click_datas)))
    train_corpus = clickdata.get_corpus(train_corpus_obj_ids_set)
    print("len(train_corpus): " + str(len(train_corpus)))
    train_samples = clickdata.get_samples(click_datas=train_click_datas, corpus=train_corpus, max_relevance=max_relevance, pos_neg_ration=POS_NEG_RATION, max_len=max_len)
    print("len(train_samples): " + str(len(train_samples)))

    # dev_click_datas, dev_corpus_obj_ids_set = clickdata.get_click_datas4bert(before_day=1, end_day=1)
    # print("len(dev_click_datas): " + str(len(dev_click_datas)))
    # dev_corpus = clickdata.get_corpus(dev_corpus_obj_ids_set)
    # print("len(dev_corpus): " + str(len(dev_corpus)))

    # dev_samples = clickdata.get_samples(
    #     dev_click_datas, dev_corpus, NUM_DEV_QUERIES, NUM_MAX_DEV_NEGATIVES)
    
    print("len(train_samples): " + str(len(train_samples)))
    # print("len(dev_samples): " + str(len(dev_samples)))

    current_dir = os.path.abspath(os.path.dirname(__file__))
    train_pkl_file = os.path.join(
        current_dir, '../data/pkl4bert/train.pkl')
    clickdata.write_samples_pkl(train_samples, train_pkl_file)

    # dev_pkl_file = os.path.join(
    #     current_dir, '../data/pkl4bert/dev.pkl')
    # clickdata.write_samples_pkl(dev_samples, dev_pkl_file)
    print("数据读取完成")
