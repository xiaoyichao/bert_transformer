# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/4/26 11:56
@Desciption : 工具类
'''
import re
from opencc import OpenCC
import unicodedata
import hashlib
import zlib
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")
from ann_engine.get_faiss_id import FaissId


# 字符大写转小写 规则
en2zhChartable = {ord(f):ord(t) for f,t in zip(
     u'，。！？【】（）％＃＠＆１２３４５６７８９０',
     u',.!?[]()%#@&1234567890')}
# 过滤表情符号 规则
filterEmoji = re.compile(u'[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]')

# 规则 过滤 note描述中 用户自己打的标签 #客厅
datepat2=re.compile('#.*? ')
datepat3_1=re.compile('[ |.|,]#[^\s,. ]+')
datepat3_2=re.compile('#[^\s,. #]+$')

class Tool :
    '''
    @Author  : xuzhongjie
    @Desciption : 工具类
    '''
    # 回答类型
    TYPE_ANSWER = 4
    # note类型
    TYPE_NOTE = 0

    def T2S(self, text):
        """
        繁体转简体 字符大写转小写
        Args:
            text:  文本

        Returns: string
        @Author  : xuzhongjie
        """
        text1 = text.translate(en2zhChartable)
        text2 = OpenCC('t2s').convert(text1)
        res = unicodedata.normalize('NFKC', text2)
        return res.strip()

    def processNoteRemarkData(self, text):
        """
        过滤 note描述中 用户自己打的标签 #客厅
        Args:
            text: 文本

        Returns: string
        @Author  : xuzhongjie
        """
        text1 = self.T2S(text)
        data2 = datepat2.sub('', text1)
        data3 = datepat3_1.sub('', data2)
        data4 = datepat3_2.sub('', data3)
        return data4

    def filter_tag(self, htmlstr):
        """
        过滤html标签 连续换行 连续空格
        Args:
            htmlstr: 文本

        Returns: string
        @Author  : xuzhongjie
        """

        re_cdata = re.compile('<!DOCTYPE HTML PUBLIC[^>]*>', re.I)
        re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # 过滤脚本
        re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # 过滤style
        re_br = re.compile('<br\s*?/?>')
        re_h = re.compile('</?\w+[^>]*>')
        re_comment = re.compile('<!--[\s\S]*-->')
        s = re_cdata.sub('', htmlstr)
        #     s = filterEmoji.sub(" ", s)
        s = re_script.sub('', s)
        s = re_style.sub('', s)
        s = re_br.sub('\n', s)
        s = re_h.sub(' ', s)
        s = self.replaceCharEntity(s)
        s = re_comment.sub('', s)
        blank_line = re.compile('\n+')
        s = blank_line.sub('\n', s)
        s = re.sub('[\f\r\t\v]', ' ', s)
        s = s.replace("…", " ")
        s = re.sub(' +', ' ', s)
        s = re.sub('[\.\。]{2,}', ' ', s)
        return s

    def replaceCharEntity(self, htmlstr):
        """
        过滤实体字符
        Args:
            htmlstr: 文本

        Returns: string
        @Author  : xuzhongjie
        """
        CHAR_ENTITIES = {'nbsp': '', '160': '',
                         'lt': '<', '60': '<',
                         'gt': '>', '62': '>',
                         'amp': '&', '38': '&',
                         'quot': '"''"', '34': '"'}
        re_charEntity = re.compile(r'&#?(?P<name>\w+);')  # 命名组,把 匹配字段中\w+的部分命名为name,可以用group函数获取
        sz = re_charEntity.search(htmlstr)
        while sz:
            # entity=sz.group()
            key = sz.group('name')  # 命名组的获取
            try:
                htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)  # 1表示替换第一个匹配
                sz = re_charEntity.search(htmlstr)
            except KeyError:
                htmlstr = re_charEntity.sub('', htmlstr, 1)
                sz = re_charEntity.search(htmlstr)
        return htmlstr

    @staticmethod
    def getLogFileName(currentLogPath, currentProjectPath):
        """
        返回以项目路径为根路径的 相对路径  例如 ： projectPath/es/Note.py  会返回  es_Note
        Args:
            currentLogPath:  当前执行文件的路径
            currentProjectPath: 项目路径

        Returns:

        """
        absLogPath = re.sub('^(' + currentProjectPath + '/)', '', currentLogPath)
        absLogPath = absLogPath.split("/")
        return "_".join(absLogPath).replace(".py", "")

    @classmethod
    def MD5(cls, beMd5Str):
        """
        md5 加密
        Args:
            beMd5Str:  被加密的数据

        Returns:

        """
        md = hashlib.md5()  # 创建md5对象
        md.update(beMd5Str.encode(encoding="utf-8"))
        return md.hexdigest()

    @classmethod
    def getTypeByObjId(cls, objId):
        """
        获取内容类型
        Args:
            objId:  内容id

        Returns:

        """
        typeStr = objId[8:9]
        return int(typeStr)

    @classmethod
    def getTableNumByUidForPhoto(self, uid):
        subTableNum = 4
        uidStr = str(uid).encode('utf8')
        uidCrc32 = zlib.crc32(uidStr)
        return (int(uidCrc32 / subTableNum)) % subTableNum

    @classmethod
    def get_dir_size(cls, dir):
        '''
        Author: xiaoyichao
        param {*}
        Description: 返回文件夹的大小，单位M
        '''
        dir_size = 0
        for root, dirs, files in os.walk(dir):
            dir_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return ((dir_size / 1024) / 1024)

    @classmethod
    def get_content_type_ids_dict(cls, add_date_obj_ids):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 构建content_type和obj_id的字典
        '''
        content_type_ids_dict = {}
        for obj_id in add_date_obj_ids:
            content_type = FaissId.get_content_type(obj_id)
            # 没有这个key就新建key-value，有则在原有value上append
            content_type_ids_dict.setdefault(content_type, []).append(obj_id)
        return content_type_ids_dict


def get_models(dir_path, reverse=True):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 返回文件夹里所有的文件路径（倒序）和最新文件路径
    '''
    model_list = sorted(os.listdir(dir_path), reverse=reverse)
    model_path_list = [os.path.join(dir_path, model_name) for model_name in model_list]
    for mode_path in model_path_list:
        if os.path.isdir(mode_path): # 是否是文件夹
            if not os.listdir(mode_path): # 是否是空文件夹
                model_path_list.remove(mode_path)
                os.rmdir(mode_path )
    return model_list, model_path_list


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



    
    
    
if __name__ == '__main__':
    print(Tool.getTableNumByUidForPhoto(11907))

