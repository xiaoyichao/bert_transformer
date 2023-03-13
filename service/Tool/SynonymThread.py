# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 同义词监测线程
'''
import asyncio
import os
import time
from common.common_log import Logger, get_log_name

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class SynonymThread():
    TYPE = "SynonymHotLoad"

    @classmethod
    async def run(self, Synonym):
        """
        动态加载同义词
        Returns:

        """
        try:
            while True:
                # 获取文件修改时间
                SynonymTagFileModifyTime = os.stat(Synonym.SynonymFilePath).st_mtime

                # 同义词 有被修改
                if SynonymTagFileModifyTime > Synonym.SynonymTagModifyTime:
                    log.logger.info(Synonym.SynonymFilePath + " 文件有修改")

                    allSynonymTagSet = set([line.strip() for line in open(
                        Synonym.SynonymFilePath, 'r', encoding='utf-8').readlines()])

                    SynonymWords = Synonym.GetSynonymWords(allSynonymTagSet)

                    # log.logger.debug(SynonymWords)

                    write_marker = Synonym.marker.gen_wlock()
                    write_marker.acquire()
                    Synonym.SynonymWords = SynonymWords
                    write_marker.release()

                    Synonym.SynonymTagModifyTime = SynonymTagFileModifyTime

                await asyncio.sleep(10)

        except Exception as e:
            err_log.logger.error(cls.TYPE + ' 线程错误', e)
            log.logger.error(cls.TYPE + ' 线程错误', e)


if __name__ == '__main__':
    from service.Tool.Synonym import Synonym

    Synonym.loadSynonym()
    print(Synonym.GetSynonymByKeywords(["客厅"]))
    cls = SynonymThread()
    cls.start()
    cls.join()
    print(Synonym.GetSynonymByKeywords(["客厅"]))
