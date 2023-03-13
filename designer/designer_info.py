# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2020-11-03 11:31:00
LastEditTime: 2021-07-05 15:13:34
Description: 设计师相关的基础数据
'''
import os
import sys
import configparser
os.chdir(sys.path[0])
sys.path.append("../")
from common import create_connection
from common.common_log import Logger, get_log_name

current_dir = os.path.abspath(os.path.dirname(__file__))
config_file = os.path.join(current_dir, '../config/config.ini')

root_config = configparser.ConfigParser()
root_config.read(config_file)

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
err_log_name = get_log_name(abs_path, "error")

log = Logger(log_name=all_log_name)
err_log = Logger(log_name=err_log_name)


class Designer(object):
    '''
    Author: xiaoyichao
    param {*} self
    Description: 设计师基础数据的类
    '''

    def get_b_class_designers(self):
        '''
        Author: xiaoyichao
        param {*}
        Description: 需要过滤的(临时)B级用户uid
        '''
        content_db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['hhzhome_admin'], 'hhzhome_admin')
        content_cursor = content_db.cursor()
        sql = ''' SELECT uid
            FROM  hhz_adm_user_type
            WHERE quality in (9,10,11)
            '''
        content_cursor.execute(sql)
        b_uid_set = set()
        for info in content_cursor:
            b_uid_set.add(info[0])
        log.logger.info("B class len:%(B_class_len)s" % {"B_class_len": len(b_uid_set)})
        content_cursor.close()
        content_db.close()
        return b_uid_set

    def basic_info(self, use_son_designer=False):
        '''
        Author: xiaoyichao
        param {*}use_son_designer=False,获取的数据是所有设计师账户，排除了子账户和测试账户
        Description: 设计师的部分基础信息。目前只返回了设计师的uid数据
        '''
        content_db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_member'], 'db_member')
        content_cursor = content_db.cursor()
        if use_son_designer:
            # 获取所有设计师账户，包含了子账户和测试账户
            sql = ''' SELECT hhz_designer.uid, hhz_member.nick, hhz_designer.min_price, hhz_designer.max_price, hhz_designer.main_area, hhz_designer.other_area, hhz_designer_auth.designer_type, hhz_designer_auth.auth_time
            FROM  hhz_designer LEFT JOIN hhz_designer_auth ON hhz_designer.uid = hhz_designer_auth.uid LEFT JOIN hhz_member ON hhz_designer.uid = hhz_member.uid
            WHERE hhz_designer_auth.auth_status = 5 AND hhz_designer.status = 1
            ORDER BY hhz_designer.uid
            '''
        else:
            # 获取所有设计师账户，排除了子账户和测试账户
            sql = '''SELECT hhz_designer.uid, hhz_member.nick, hhz_designer.min_price, hhz_designer.max_price, hhz_designer.main_area, hhz_designer.other_area, hhz_designer_auth.designer_type, hhz_designer_auth.auth_time
            FROM  hhz_designer LEFT JOIN hhz_designer_auth ON hhz_designer.uid = hhz_designer_auth.uid LEFT JOIN hhz_member ON hhz_designer.uid = hhz_member.uid
            WHERE hhz_designer_auth.auth_status = 5 AND  hhz_designer_auth.designer_type !=4  AND hhz_designer_auth.is_test = 0 AND hhz_designer.status = 1
            ORDER BY hhz_designer.uid  '''

        content_cursor.execute(sql)
        self.designers_uid_list = []

        for info in content_cursor:
            uid = info[0]
            self.designers_uid_list.append(uid)
        content_cursor.close()
        content_db.close()
        return self.designers_uid_list

    def team_data_relationship(self):
        '''
        Author: xiaoyichao
        param {*}
        Description: 设计师与团队的绑定关系
        '''
        content_db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_member'], 'db_member')
        content_cursor = content_db.cursor()

        sql_team = ''' SELECT hhz_designer_team.org_uid, hhz_designer_team.member_uid FROM  hhz_designer_team WHERE hhz_designer_team.status = 1 '''
        content_cursor.execute(sql_team)
        self.sub_designers = set()
        self.org_sub_relationship = {}
        for info in content_cursor:
            org_uid = info[0]
            member_uid = info[1]
            self.sub_designers.add(member_uid)
            self.org_sub_relationship[member_uid] = org_uid
        content_cursor.close()
        content_db.close()
        return self.sub_designers, self.org_sub_relationship

    def test_designers(self):
        '''
        Author: xiaoyichao
        param {*}
        Description: 获取内部测试的设计师账号
        '''
        content_db = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_member'], 'db_member')
        content_cursor = content_db.cursor()
        sql_test_uids = ''' SELECT uid FROM  hhz_designer_auth WHERE  is_test = 1 '''
        content_cursor.execute(sql_test_uids)
        self.test_designer_uids = set()
        for info in content_cursor:
            self.test_designer_uids.add(info[0])
        content_cursor.close()
        content_db.close()
        return self.test_designer_uids

    def get_designer_list(self, use_son_designer=False):
        '''
        @Author: xiaoyichao
        @param {*}
        @Description: 获取设计师的uid, 注意可以剔除子账户。测试账户已经被剔除了，不可选择。
        '''
        designers_uid_list = self.basic_info(use_son_designer)
        return designers_uid_list


if __name__ == "__main__":
    designer = Designer()
    designers_uid_list = designer.get_designer_list()
    print(designers_uid_list)
