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

class JiebaThreadTest():

    TYPE = "JiebaHotLoadTest"

    @classmethod
    async def run(cls, JiebaHHZTest):
        """
        动态加载词库
        Returns:

        """
        try:
            while True:
                # 获取文件修改时间
                AdminTagFileModifyTime = os.stat(JiebaHHZTest.AdminTagFile).st_mtime
                UserTagFileModifyTime = os.stat(JiebaHHZTest.UserTagFile).st_mtime

                # admin tag 有被修改
                if AdminTagFileModifyTime > JiebaHHZTest.AdminTagModifyTime:
                    log.logger.info(JiebaHHZTest.AdminTagFile + " 文件有修改")
                    allAdminTagSet = set([line.strip() for line in open(
                        JiebaHHZTest.AdminTagFile, 'r', encoding='utf-8').readlines()])

                    AdminTagAdd = allAdminTagSet - JiebaHHZTest.AllAdminTagSet
                    AdminTagDelete = JiebaHHZTest.AllAdminTagSet - allAdminTagSet

                    log.logger.info("新增标签：" + ",".join(AdminTagAdd) + ", 删除标签：" + ",".join(AdminTagDelete))

                    JiebaHHZTest.AllAdminTagSet = JiebaHHZTest.AllAdminTagSet | AdminTagAdd
                    JiebaHHZTest.AllAdminTagSet = JiebaHHZTest.AllAdminTagSet - AdminTagDelete
                    JiebaHHZTest.dynamicAddWords(AdminTagAdd)
                    JiebaHHZTest.dynamicDeleteWords(AdminTagDelete)

                    JiebaHHZTest.AdminTagModifyTime = AdminTagFileModifyTime

                if UserTagFileModifyTime > JiebaHHZTest.UserTagModifyTime:
                    log.logger.info(JiebaHHZTest.UserTagFile + " 文件有修改")

                    allUserTagSet = set([line.strip() for line in open(
                        JiebaHHZTest.UserTagFile, 'r', encoding='utf-8').readlines()])
                    UserTagAdd = allUserTagSet - JiebaHHZTest.AllUserTagSet
                    UserTagDelete = JiebaHHZTest.AllUserTagSet - allUserTagSet

                    log.logger.info("新增自定义词汇：" + ",".join(UserTagAdd) + ", 删除自定义词汇：" + ",".join(UserTagDelete))
                    JiebaHHZTest.AllUserTagSet = JiebaHHZTest.AllUserTagSet | UserTagAdd
                    JiebaHHZTest.AllUserTagSet = JiebaHHZTest.AllUserTagSet - UserTagDelete
                    JiebaHHZTest.dynamicAddWords(UserTagAdd)
                    JiebaHHZTest.dynamicDeleteWords(UserTagDelete)

                    JiebaHHZTest.UserTagModifyTime = UserTagFileModifyTime

                await asyncio.sleep(10)

        except Exception as e:
            err_log.logger.error(cls.TYPE + ' 线程错误', e)
            log.logger.error(cls.TYPE + ' 线程错误', e)


if __name__ == '__main__':
    from service.Tool.JiebaHHZTest import JiebaHHZTest

    print(JiebaHHZTest.SplitWords("测试暗黑者你好呀"))
