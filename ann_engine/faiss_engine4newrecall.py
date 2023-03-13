# coding=UTF-8
'''
@Author: xiaoyichao
LastEditors: xiaoyichao
@Date: 2020-06-19 17:14:35
LastEditTime: 2022-08-02 14:58:41
@Description: 提供faiss建立索引，索引增量添加数据，检索数据的功能。注意faiss不支持float64，最大精度floa32。
'''
import os
os.environ['TF_KERAS'] = '1'
import time
import codecs
import faiss
import math
import torch
import configparser
import numpy as np
import sys
os.chdir(sys.path[0])
sys.path.append("../")
from common.common_log import Logger, get_log_name
from rw_vector.vecbinNewRecall import vecbinNewRecall
from common.get_ip import get_host_ip
from keras_bert import Tokenizer

dir_name = os.path.abspath(os.path.dirname(__file__))

faiss_models_dir = "/data/search_opt_model/faiss_model4newrecall"

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

config_file = os.path.join(dir_name, '../config/config.ini')
root_config = configparser.ConfigParser()
root_config.read(config_file)
save_model_nums = int(root_config["faiss"]['save_model_nums'])  # 最多保存几个faiss的模型

log = Logger(log_name=all_log_name)
vecbinNewRecallCls = vecbinNewRecall()



def get_models():
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 返回faiss model 文件夹里所有的文件路径（倒序）和最新文件路径
    '''
    model_list = sorted(os.listdir(faiss_models_dir), reverse=True)
    model_path_list = [os.path.join(faiss_models_dir, model_name) for model_name in model_list]
    newset_model = model_path_list[0]
    return model_path_list, newset_model


class ANNEngine(object):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 提供faiss建立索引，索引增量添加数据的功能。注意faiss不支持float64，最大精度floa32. 
    '''

    def __init__(self, nprobe=20, use_gpu="auto", *args, **kwargs):
        '''
        @Author: xiaoyichao
        @param {nprobe: 每次查询需要查的倒排list的个数, *args：任意多个位置参数（无名参数），**kwargs：任意多个有名参数（关键词参数）}
        @Description: 初始化
        '''
        if torch.cuda.is_available():
            try:
                # 如果容器的GPU 和CUDA 的匹配有问题，这句话会失败，这会直接让程序kill,所以先检测有没有GPU
                ngpus = faiss.get_num_gpus()
            except Exception:
                log.logger.info("faiss查询GPU资源失败，默认使用CPU")
                ngpus = 0
        else:
            ngpus = 0

        if use_gpu == "auto":
            if ngpus > 0:
                self.use_gpu = True
                log.logger.info("train engineer选择了auto参数，服务器安装了faiss-gpu，并且有可用的GPU资源")
            else:
                self.use_gpu = False
                log.logger.info("train engineer选择了auto参数，但是服务器没有安装faiss-gpu，或者没有可用的GPU资源")
        else:
            if use_gpu == 1 and ngpus > 0:
                self.use_gpu = True
                log.logger.info("train engineer选择了使用了GPU")
            else:
                self.use_gpu = False
                log.logger.info("train engineer选择了使用了CPU")
        
        self.nprobe = nprobe
        self.use_gpu = False

    def train(self, obj_vecs: np.array, obj_ids: np.array, file_time):
        """
        Author: xiaoyichao
        [初次建立索引]

        Args:
            obj_vecs ([nplist]): np[多个内容的向量组成np_list]
            obj_ids ([nplist]): np[内容对应的obj_passge_id]
            注意faiss的输入和输出都是array.
        """
        nlist = int(math.sqrt(obj_vecs.shape[0])) * 2  # 聚类中心
        dimension = int(obj_vecs.shape[1])  # 维度


        """IndexFlatIP=点乘，归一化的向量点乘即cosine相似度, IndexFlatL2=L2距离，即欧式距离"""
        quantizer = faiss.IndexFlatIP(dimension)

        if self.use_gpu:  # 对量化器进行转化
            quantizer = faiss.index_cpu_to_all_gpus(quantizer)  # 使用全部GPU
            # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 只使用一块GPU
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        index.train(obj_vecs)
        # index.add(obj_vecs) # 自动索引id
        index.add_with_ids(obj_vecs, obj_ids)  # 使用自定义的索引id,但是只支持int64
        index.nprobe = self.nprobe

        """faiss模型的文件名是faiss+时间戳"""
        faiss_index_path = os.path.join(
            faiss_models_dir, 'faiss_%s.index' % str(int(file_time)))

        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)
            log.logger.info("将GPU index model 转化为CPU index model")
        faiss.write_index(index, faiss_index_path)
        log.logger.info("保存CPU index model：%s" % faiss_index_path)

    def train4recall(self, obj_vecs: np.array, obj_ids: np.array):
        """
        Author: xiaoyichao
        [初次建立索引]

        Args:
            obj_vecs ([nplist]): np[多个内容的向量组成np_list]
            obj_ids ([nplist]): np[内容对应的obj_passge_id]
            注意faiss的输入和输出都是array.
        """
        obj_vecs = obj_vecs.astype(np.float32)
        nlist = int(math.sqrt(obj_vecs.shape[0])) * 2  # 聚类中心
        dimension = int(obj_vecs.shape[1])  # 维度
        obj_ids_int = np.array([i for i in range(len(obj_ids))])

        """IndexFlatIP=点乘，归一化的向量点乘即cosine相似度, IndexFlatL2=L2距离，即欧式距离"""
        quantizer = faiss.IndexFlatIP(dimension)

        if self.use_gpu:  # 对量化器进行转化
            log.logger.info("faiss 使用 GPU （显卡）")
            quantizer = faiss.index_cpu_to_all_gpus(quantizer)  # 使用全部GPU
            # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 只使用一块GPU
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            log.logger.info("faiss 使用 CPU ")
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        index.train(obj_vecs)
        # index.add(obj_vecs) # 自动索引id
        index.add_with_ids(obj_vecs, obj_ids_int)  # 使用自定义的索引id,但是只支持int64
        index.nprobe = self.nprobe

        """faiss模型的文件名是faiss+时间戳"""
        faiss_index_path = os.path.join(
            faiss_models_dir, 'faiss_%s.index' % str(int(time.time())))

        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)
            log.logger.info("将GPU index model 转化为CPU index model")
        faiss.write_index(index, faiss_index_path)
        log.logger.info("保存CPU index model：%s" % faiss_index_path)
        
    def increment(self, obj_vecs: np.array, obj_ids: np.array):
        """
        Author: xiaoyichao
        [读取最新的索引文件，增量构建索引，写入新的索引文件]
        Args:
            parameter_list ([type]): [description]
        """
        model_path_list, newset_model = get_models()
        index = faiss.read_index(newset_model)
        if self.use_gpu:
            try:
                index = faiss.index_cpu_to_all_gpus(index)
            except Exception:
                log.logger.error("将cpu index 转化为 gpu index 失败！！！")
                Logger(log_name=err_log_name).logger.error('将cpu index 转化为 gpu index 失败！！！')
        log.logger.info('读取最近的模型：%s' % newset_model)
        index.add_with_ids(obj_vecs, obj_ids)
        faiss_index_path = os.path.join(
            faiss_models_dir, 'faiss_%s.index' % str(int(time.time())))
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, faiss_index_path)
        log.logger.info("faiss文件写入成功：" % faiss_index_path)

        log.logger.info('写入新的模型：%s' % faiss_index_path)
        if len(model_path_list) > save_model_nums:   # 如果文件数量超过save_model_nums个，删除最新的save_model_nums个之外的老文件，只保留最近的save_model_nums个model
            for model_path in model_path_list[save_model_nums-1:]:
                os.remove(model_path)
                log.logger.info('删除旧的模型：%s' % model_path)


class SearchANN(object):
    """
    [faiss检索功能的类]

    Args:
        object ([type]): [description]
    """
    def __init__(self, use_gpu="auto"):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 初始化，读取最新的faiss model
        '''
        _, newset_model = get_models()
        self.index = faiss.read_index(newset_model)

        # 获取 内容 id 对应 index
        def getIndex2ObjId():
            from service.Init.InitResource import InitResource
            from db.hhzSearch.ObjVecId2Index import ObjVecId2Index

            InitResourceCls = InitResource()
            InitResourceCls.initDbSearchConnect()

            index2ObjId = ObjVecId2Index.getAllOIndex2ObjId(InitResourceCls)
            InitResourceCls.closeDbSearchConnect()

            return index2ObjId

        self.int_id_dict = vecbinNewRecallCls.load_get_id_vec_dict()
        self.index2ObjId = getIndex2ObjId()

        if use_gpu == "auto":   
            if torch.cuda.is_available():
                try:
                    ngpus = faiss.get_num_gpus()
                except Exception:
                    log.logger.info("faiss查询GPU资源失败，默认使用CPU")
                    ngpus = 0
            else:
                ngpus = 0
            # ngpus = 0
            if ngpus > 0:
                self.use_gpu = True
                self.index = faiss.index_cpu_to_all_gpus(self.index)
                log.logger.info("search ngineer选择了auto参数，服务器安装了faiss-gpu，并且有可用的GPU资源")
            else:
                log.logger.info("search engineer选择了auto参数，但是服务器没有安装faiss-gpu，或者没有可用的GPU资源")
        else:
            if use_gpu == 1:
                self.use_gpu = True
                try:
                    self.index = faiss.index_cpu_to_all_gpus(self.index)
                except Exception:
                    log.logger.error("将cpu index 转化为 gpu index 失败！！！")
                    Logger(log_name=err_log_name).logger.error('将cpu index 转化为 gpu index 失败！！！')
                log.logger.info("search engineer选择了使用了GPU")
            else:
                self.use_gpu = False
                log.logger.info("search engineer选择了使用了CPU")
        self.use_gpu = False

    def search(self, query_vec, search_length):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: faiss的检索功能，
        需要检索的数据需要包装这样的array[array[]]数据格式，也就是可以同时对很对数据进行检索，得出对应的答案, 输出的数据也是list
        '''
        '''直接float64输入进去也可以，faiss会自己把精度降低到float32'''
        # sims, ids = self.index.search(
        #     query_vec.astype('float32'), search_length)
        # 如search_length=0,则直接跳过faiss的召回
        if search_length == 0:
            return [], []
        query_vec = query_vec.astype('float32')
        query_vec = np.array([query_vec])  # 允许输入多个向量，所以数据需要构造成2层的

        sims, int_ids = self.index.search(query_vec, search_length)

        int_ids = int_ids[0].tolist()

        obj_ids =  [self.index2ObjId[str(int_id)] for int_id in int_ids]
        sims = sims[0].tolist()
        log.logger.debug('int_ids: %s' % str(int_ids))
        log.logger.debug('obj_ids: %s' % str(obj_ids))
        return sims, obj_ids

'''检索'''
if __name__ == "__main__":
    import json
    import requests

    tf_serving_url = "http://127.0.0.1:8521/v1/models/bert_dssm_query:predict"  # 使用最新的模型

    token_dict = {}
    with codecs.open("/data3/xuzhongjie/chinese_L-12_H-768_A-12/vocab.txt", "r", "utf-8") as fr:
        for line in fr:
            line = line.strip()
            token_dict[line] = len(token_dict)
    tokenizer = Tokenizer(token_dict)

    max_len = 64
    qt, qs = tokenizer.encode("沙发", max_len=max_len)
    features = [{
        "query_input_token": qt,
        "query_input_segment": qs,
    }]

    data = json.dumps({"signature_name": "serving_default", "instances": features})
    headers = {"content-type": "application/json"}
    json_response = requests.post(tf_serving_url, data=data, headers=headers)

    json_data = json.loads(json_response.text)

    predictions = json_data["predictions"]
    query_vec = predictions[0]
    # print("=========query_vec===========", query_vec)

    # doc
    # tf_serving_url = "http://172.16.10.10:8520/v1/models/bert_dssm_doc:predict"  # 使用最新的模型
    #
    # tokenizer = Tokenizer(token_dict)
    # max_len = 64
    # qt, qs = tokenizer.encode("这个客厅是小户型的呀", max_len=max_len)
    # print("==================", qt, qs, "================")
    # features = [{
    #     "pos_doc_input_token": qt,
    #     "pos_doc_input_segment": qs,
    # }]
    #
    # data = json.dumps({"signature_name": "serving_default", "instances": features})
    # headers = {"content-type": "application/json"}
    # json_response = requests.post(tf_serving_url, data=data, headers=headers)
    #
    # json_data = json.loads(json_response.text)
    #
    # predictions = json_data["predictions"]
    # doc_vec = predictions[0]
    #
    # print("======query_vec========", query_vec)
    # print("======doc_vec========", doc_vec)
    # from service.Weight.calc_weight import CalcWeight
    # import numpy as np
    #
    # print(CalcWeight.cosine_sim_hhz(np.array(query_vec), np.array(doc_vec)))


    # print("================= query_vec", query_vec, "query_vec ===================")
    # import  sys
    # sys.exit(0)
    search_ann = SearchANN()
    sims, obj_ids = search_ann.search(np.array(predictions[0], dtype="float32"), 10)
    #
    print(sims)
    print(obj_ids)

   