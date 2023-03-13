# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2020-10-22 15:44:46
LastEditTime: 2021-07-22 17:25:52
Description: 
'''
import os
import sys
import re
import json
os.chdir(sys.path[0])
sys.path.append("../")


dir_name = os.path.abspath(os.path.dirname(__file__))
word_dic_path = "/home/adm_rsync_dir/search/word.dic"
user_dic_path = os.path.join(dir_name, "../data/userdict.txt")
term_weight_backstage_path = os.path.join(dir_name, "../data/term_weight_backstage/term_weight.json")
tmp_user_dict_path = os.path.join(dir_name, "../data/hhz_user_dict/tmp_user_dict.txt")
user_dict_path = os.path.join(dir_name, "../data/hhz_user_dict/user_dict.txt")
query_need_cut_path = os.path.join(dir_name, "../data/hhz_user_dict/query_need_cut.txt")


def clean_text(sentence):
    """
    [对文本数据进行清洗，清除html标签和url]

    Args:
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    url_reg = r'[a-z]*[:.]+\S+'
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, '', sentence)  # 去除html标签
    sentence = re.sub(url_reg, '', sentence)
    sentence = sentence.replace('\ufeff', ' ')
    sentence = sentence.replace('\xa0', ' ')
    return sentence


class DataInfo(object):
    '''
    Author: xiaoyichao
    param {type}
    Description: 获取各种训练数据的类
    '''

    def __init__(self):
        pass

    def process_dic(self):
        '''
        Author: xiaoyichao
        param {type}
        Description: 读取搜索项目使用的词典文件，生成指定格式的词典文件
        '''
        words = []
        with open(word_dic_path, "r") as f:
            lines = f.read()
            lines = lines.split("\n")
            for line in lines:
                word = line.split(" ")[0]
                words.append(word)
        with open(user_dic_path, "w") as u:
            for word in words:
                u.write(word+"\n")
        print("写入词典成功，单词数量：", len(words))

    @classmethod
    def process_term_weight(cls):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description:  读取词权重的后台数据
        '''
        with open(term_weight_backstage_path, 'r') as load_f:
            term_weight_dict = json.load(load_f)
        return term_weight_dict

    @classmethod
    def get_need_cut_query(cls):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description:  得到需要分词的query组成的list
        '''
        need_cut_query_list = []
        with open(tmp_user_dict_path, 'r') as load_f:
            lines = load_f.read()
            lines = lines.split("\n")
        for line in lines:
            line = line.split(",")
            if len(line) > 1:
                line = "".join(line)
                need_cut_query_list.append(line)
        return need_cut_query_list


    @classmethod
    def write_need_cut_query2file(cls):
        need_cut_query_list = cls.get_need_cut_query()
        with open(query_need_cut_path, "w") as f:
            for query in set(need_cut_query_list):
                f.write(query+"\n")

    @classmethod
    def read_queries(cls):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description:  读取临时的自有词表
        '''
        query_list = []
        with open(tmp_user_dict_path, 'r') as load_f:
            lines = load_f.read()
            lines = lines.split("\n")
        for line in lines:
            line = line.split(",")
            query_list = query_list + line
        return query_list

    @classmethod
    def write_user_dict2file(cls):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description:  将临时的自有词表数据写入文件
        '''
        query_list = cls.read_queries()
        with open(user_dict_path, "w") as f:
            for word in set(query_list):
                f.write(word+"\n")


if __name__ == "__main__":
    datainfo = DataInfo()
    datainfo.get_need_cut_query()
    datainfo.write_user_dict2file()
    datainfo.write_need_cut_query2file()
