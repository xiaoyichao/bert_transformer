# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/10 15:49
@Desciption : 同义词转换类
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")
from common.common_log import Logger, get_log_name

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

from readerwriterlock import rwlock
from common.tool import Tool

class SynonymTest(object):
    # 同义词文件
    # SynonymFilePath = "/home/adm_rsync_dir/search_op/search_synonyms.txt"
    SynonymFilePath = "/home/adm_rsync_dir/search_op/search_synonyms_test.txt"

    # 同义词汇 修改时间
    SynonymTagModifyTime = 0

    # 处理后的同义词字典
    SynonymWords = {}

    # 读写锁  防止 资源抢占
    marker = rwlock.RWLockFair()

    @classmethod
    def loadSynonym(cls):
        """
        加载同义词到内存
        """
        allSynonymWords = set([line.strip() for line in open(
            cls.SynonymFilePath, 'r', encoding='utf-8').readlines()])

        cls.SynonymWords = cls.GetSynonymWords(allSynonymWords)

        cls.SynonymTagModifyTime = os.stat(cls.SynonymFilePath).st_mtime

        # log.logger.debug(cls.SynonymWords)

    @classmethod
    def GetSynonymWords(cls, allSynonymWords):
        """
        根据同义词文件词汇格式 格式化成 同义词字典格式
        """
        synonym_words_hash = {}

        # 别名
        alias_hash = {}
        # 上下级
        subordinate_hash = {}

        for SynonymWord in allSynonymWords:
            # 工具类
            ToolCLs = Tool()
            SynonymWord = ToolCLs.T2S(SynonymWord)
            synonym_word_info = SynonymWord.split("=>")

            if len(synonym_word_info) == 2:
                keyword = synonym_word_info[0]
                synonym_words_str = synonym_word_info[1]

                synonym_words_str = synonym_words_str.strip()
                synonym_words_tmp = set(synonym_words_str.split(","))

                keywordSet = set(keyword.split(","))
                for keyword in keywordSet:
                    keywordTmp = [keyword]
                    synonym_words_tmp = synonym_words_tmp - set(keywordTmp)

                    synonym_words = set()
                    for synonym_words_tmp_item in synonym_words_tmp:
                        synonym_words_tmp_item = synonym_words_tmp_item.strip()
                        if len(synonym_words_tmp_item) > 0:
                            synonym_words.add(synonym_words_tmp_item)

                    if len(synonym_words) > 0:
                        subordinate_hash[keyword] = synonym_words
            elif len(synonym_word_info) == 1:
                alias = synonym_word_info[0]
                aliasList = alias.split(",")

                aliasSet = set()
                for aliasTmp in aliasList:
                    aliasTmp = aliasTmp.strip()
                    aliasSet.add(aliasTmp)

                for aliasTmp2 in aliasList:
                    aliasListTmp = [aliasTmp2]
                    alias_hash[aliasTmp2] = aliasSet - set(aliasListTmp)

        alias_key = set(alias_hash.keys())
        subordinate_key = set(subordinate_hash.keys())

        all_keys = alias_key | subordinate_key

        for key in all_keys:
            synonym_words_set = set()

            if key in alias_hash:
                synonym_words_set = synonym_words_set | alias_hash[key]

            if key in subordinate_hash:
                synonym_words_set = synonym_words_set | subordinate_hash[key]

            if len(synonym_words_set) > 0:
                synonym_words_hash[key] = synonym_words_set

        return synonym_words_hash

    @classmethod
    def GetSynonymByKeywords(cls, keywords):
        """
        获取关键词的同义词
        """
        SynonymWords = set()

        read_marker = cls.marker.gen_rlock()

        read_marker.acquire()
        for keyword in keywords:
            if keyword in cls.SynonymWords:
                SynonymWords = SynonymWords | cls.SynonymWords[keyword]
        read_marker.release()

        return SynonymWords

if __name__ == "__main__":
    SynonymTest.loadSynonym()


