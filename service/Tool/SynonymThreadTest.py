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

class SynonymThreadTest():
    TYPE = "SynonymHotLoadTest"

    @classmethod
    async def run(self, SynonymTest):
        """
        动态加载同义词
        Returns:

        """
        try:
            while True:
                # 获取文件修改时间
                SynonymTagFileModifyTime = os.stat(SynonymTest.SynonymFilePath).st_mtime

                # 同义词 有被修改
                if SynonymTagFileModifyTime > SynonymTest.SynonymTagModifyTime:
                    log.logger.info(SynonymTest.SynonymFilePath + " 文件有修改")

                    allSynonymTagSet = set([line.strip() for line in open(
                        SynonymTest.SynonymFilePath, 'r', encoding='utf-8').readlines()])

                    SynonymWords = SynonymTest.GetSynonymWords(allSynonymTagSet)

                    # log.logger.debug(SynonymWords)

                    write_marker = SynonymTest.marker.gen_wlock()
                    write_marker.acquire()
                    SynonymTest.SynonymWords = SynonymWords
                    write_marker.release()

                    SynonymTest.SynonymTagModifyTime = SynonymTagFileModifyTime

                await asyncio.sleep(10)

        except Exception as e:
            err_log.logger.error(cls.TYPE + ' 线程错误', e)
            log.logger.error(cls.TYPE + ' 线程错误', e)


if __name__ == '__main__':
    from service.Tool.SynonymTest import SynonymTest

    SynonymTest.loadSynonym()
    print(SynonymTest.GetSynonymByKeywords(["客厅"]))
    cls = SynonymThreadTest()
    cls.start()
    cls.join()
    print(SynonymTest.GetSynonymByKeywords(["客厅"]))
