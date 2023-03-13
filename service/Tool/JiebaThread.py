# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : jieba分词 动态加载词库的线程
'''
import os
from common.common_log import Logger, get_log_name
import asyncio

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class JiebaThread():

    TYPE = "JiebaHotLoad"

    @classmethod
    async def run(cls, JiebaHHZ):
        """
        动态加载词库
        Returns:

        """
        try:
            while True:
                # 获取文件修改时间
                AdminTagFileModifyTime = os.stat(JiebaHHZ.AdminTagFile).st_mtime
                UserTagFileModifyTime = os.stat(JiebaHHZ.UserTagFile).st_mtime

                # admin tag 有被修改
                if AdminTagFileModifyTime > JiebaHHZ.AdminTagModifyTime:
                    log.logger.info(JiebaHHZ.AdminTagFile + " 文件有修改")
                    allAdminTagSet = set([line.strip() for line in open(
                        JiebaHHZ.AdminTagFile, 'r', encoding='utf-8').readlines()])

                    AdminTagAdd = allAdminTagSet - JiebaHHZ.AllAdminTagSet
                    AdminTagDelete = JiebaHHZ.AllAdminTagSet - allAdminTagSet

                    log.logger.info("新增标签：" + ",".join(AdminTagAdd) + ", 删除标签：" + ",".join(AdminTagDelete))

                    JiebaHHZ.AllAdminTagSet = JiebaHHZ.AllAdminTagSet | AdminTagAdd
                    JiebaHHZ.AllAdminTagSet = JiebaHHZ.AllAdminTagSet - AdminTagDelete
                    JiebaHHZ.dynamicAddWords(AdminTagAdd)
                    JiebaHHZ.dynamicDeleteWords(AdminTagDelete)

                    JiebaHHZ.AdminTagModifyTime = AdminTagFileModifyTime

                if UserTagFileModifyTime > JiebaHHZ.UserTagModifyTime:
                    log.logger.info(JiebaHHZ.UserTagFile + " 文件有修改")

                    allUserTagSet = set([line.strip() for line in open(
                        JiebaHHZ.UserTagFile, 'r', encoding='utf-8').readlines()])
                    UserTagAdd = allUserTagSet - JiebaHHZ.AllUserTagSet
                    UserTagDelete = JiebaHHZ.AllUserTagSet - allUserTagSet

                    log.logger.info("新增自定义词汇：" + ",".join(UserTagAdd) + ", 删除自定义词汇：" + ",".join(UserTagDelete))
                    JiebaHHZ.AllUserTagSet = JiebaHHZ.AllUserTagSet | UserTagAdd
                    JiebaHHZ.AllUserTagSet = JiebaHHZ.AllUserTagSet - UserTagDelete
                    JiebaHHZ.dynamicAddWords(UserTagAdd)
                    JiebaHHZ.dynamicDeleteWords(UserTagDelete)

                    JiebaHHZ.UserTagModifyTime = UserTagFileModifyTime

                await asyncio.sleep(10)

        except Exception as e:
            err_log.logger.error(cls.TYPE + ' 线程错误', e)
            log.logger.error(cls.TYPE + ' 线程错误', e)


if __name__ == '__main__':
    from service.Tool.JiebaHHZ import JiebaHHZ

    print(JiebaHHZ.SplitWords("测试暗黑者你好呀"))
