# coding=UTF-8
'''
@Author: xuzhongjie
@Date: 2020-03-24 13:25:41
LastEditTime: 2021-05-13 14:46:06
@Description: 分词的相关操作
'''
import jieba
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../../")

dirName = os.path.abspath(os.path.dirname(__file__))

stopwordsFile = os.path.join(dirName, '../data/stopwords.txt')

class JiebaHHZTest(object):
    # jieba类
    JibaCls = jieba.Tokenizer()
    # 停止词
    StopWords = set()
    # admin tag
    AllAdminTagSet = set()
    # 自定义词汇
    AllUserTagSet = set()
    # admin tag 文件地址
    AdminTagFile = "/home/adm_rsync_dir/tags/tags_all"
    # admin tag 修改时间
    AdminTagModifyTime = 0

    #  自定义词汇 文件地址
    UserTagFile = "/home/adm_rsync_dir/search_op/jieba_user_define_words_test.dic"
    # 自定义词汇 修改时间
    UserTagModifyTime = 0

    #  删除 jieba 默认添加的词汇
    DelJiebaDefaultWordsFile = "/home/adm_rsync_dir/search_op/del_jieba_default_words.dic"

    @classmethod
    def StopwordsSet(cls, filepath):
        """
        停用词集合
        Args:
            filepath: 停用词文件地址

        Returns:

        """
        stopwords = set([line.strip() for line in open(
            filepath, 'r', encoding='utf-8').readlines()])
        return stopwords

    @classmethod
    def SplitWords(cls, sentence, isFilterStopwords=False):
        """
        将句子变成切词后的list
        Args:
            sentence: string 待切词文本
            isFilterStopwords: bool 是否过滤停止词

        Returns:

        """
        sentenceTerms = list(cls.JibaCls.cut(sentence))

        returnDataTmp = []

        # 过滤停止词
        if isFilterStopwords:
            if len(cls.StopWords) == 0:
                cls.StopWords = cls.StopwordsSet(stopwordsFile)

            for word in sentenceTerms:
                if word not in cls.StopWords:
                    returnDataTmp.append(word)

        else:
            returnDataTmp = sentenceTerms

        returnData = []
        for returnDataItem in returnDataTmp:
            returnDataItemTmp = returnDataItem.strip()
            if len(returnDataItemTmp) > 0 and returnDataItemTmp not in returnData:
                returnData.append(returnDataItemTmp)

        return returnData

    @classmethod
    def split_words_space(cls, sentence, isFilterStopwords=False):
        """
        将句子变成切词后的list
        Args:
            sentence: string 待切词文本
            isFilterStopwords: bool 是否过滤停止词

        Returns:

        """
        sentenceTerms = list(cls.JibaCls.cut(sentence))
        returnData = ""

        # 过滤停止词
        if isFilterStopwords:
            if len(cls.StopWords) == 0:
                cls.StopWords = cls.StopwordsSet(stopwordsFile)
            sentenceTerms_out_stopwords = []
            for word in sentenceTerms:
                if word not in cls.StopWords:
                    sentenceTerms_out_stopwords.append(word)
            returnData =' '.join(sentenceTerms_out_stopwords)
        else:
            returnData = ' '.join(sentenceTerms)

        return returnData

    @classmethod
    def dynamicAddWords(cls, words):
        """
        动态添加的词汇列表
        Args:
            words: list 待添加的词汇列表

        Returns:

        """
        if len(words) > 0:
            for word in words:
                cls.JibaCls.add_word(word)

    @classmethod
    def dynamicDeleteWords(cls, words):
        """
        动态删除的词汇列表
        Args:
            words: list 待删除的词汇列表

        Returns:

        """
        if len(words) > 0:
            for word in words:
                cls.JibaCls.del_word(word)

    @classmethod
    def del_jieba_default_words(cls):
        """
        删除jieba 默认的词汇列表
        Args:

        Returns:

        """
        del_words_set = set([line.strip() for line in open(
            cls.DelJiebaDefaultWordsFile, 'r', encoding='utf-8').readlines()])

        for word in del_words_set:
            if len(word) > 0:
                cls.JibaCls.del_word(word)

    @classmethod
    def loadUserDict(cls):
        """
        加载 admin tag 和 user tag文件，用于第一次加载词汇以及记录文件修改时间
        Returns:

        """
        cls.JibaCls.load_userdict(cls.AdminTagFile)
        cls.JibaCls.load_userdict(cls.UserTagFile)

        allAdminTagSet = set([line.strip() for line in open(
            cls.AdminTagFile, 'r', encoding='utf-8').readlines()])
        allUserTagSet = set([line.strip() for line in open(
            cls.UserTagFile, 'r', encoding='utf-8').readlines()])

        cls.AllAdminTagSet = allAdminTagSet
        cls.AllUserTagSet = allUserTagSet

        cls.AdminTagModifyTime = os.stat(cls.AdminTagFile).st_mtime
        cls.UserTagModifyTime = os.stat(cls.UserTagFile).st_mtime

if __name__ == '__main__':
    line = "智能马桶怎么选测试上呀"
    print(JiebaHHZTest.SplitWords(line))
