# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/5 14:26
@Desciption :  初始化 和 关闭  连接资源 例如 es redis db
'''
import os
import sys
import socket
from common.common_log import Logger, get_log_name
from elasticsearch import Elasticsearch
import configparser
from common import create_connection

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

dir_name = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(dir_name, "../../config/config.ini")

root_config = configparser.ConfigParser()
root_config.read(config_path)

os.chdir(sys.path[0])
sys.path.append("../../")

es_config = configparser.ConfigParser()
es_config.read("/home/resource_config/search_opt/es.ini")
es_server_ip_port = es_config["Es_config"]["es_server_ip_port"]
host_name = socket.gethostname()


search_opt_common_config_file = "/home/resource_config/search_opt/common.ini"
search_opt_common_config = configparser.ConfigParser()
search_opt_common_config.read(search_opt_common_config_file)
use_gray_redis = False if search_opt_common_config["which_redis"]["use_gray_redis"] == "False" else True


http_auth_user_name = es_config["Es_config"]["http_auth_user_name"]
http_auth_password = es_config["Es_config"]["http_auth_password"]


class InitResource(object):
    # es 连接池
    ES_CONNECT = None
    # 数据库 admin 连接池
    DB_ADMIN_CONNECT = None
    # 数据库 search 连接池
    DB_SEARCH_CONNECT = None
    # redis search content pool
    RD_SEARCH_CONTENT_CONNECTIONPOOL = None
    # redis search content pool
    RD_SEASON_CONTENT_CONNECTIONPOOL = None
    # redis search content write
    RD_SEARCH_CONTENT_WRITE = None
    # redis user_interact_ids 用户 有效点击 用户收藏
    RD_USER_VALID_CLICK_AND_FAVORITE_WRITE = None
    # redis 用户分群信息
    RD_USER_PROFILE_WRITE = None
    # es result cache
    RD_ES_RESULT_WRITE = None

    # 标签库
    DB_HHZ_TAG_CONNECT = None
    # store库
    DB_STORE_CONNECT = None

    def initResourceForVecRecall(self):
        """
        初始化连接资源 在 向量召回场景
        Args:

        Returns:

        """
        # 初始化 rd_search文件下 的 search_content 段
        self.initRdSearchContentConnect()

    def closeResourceForVecRecall(self):
        """
        关闭连接资源 在 向量召回场景
        Args:

        Returns:

        """
        # 关闭 redis search content 连接
        self.closeRdSearchContentConnect()

    def initResource(self):
        """
        初始化连接资源
        Args:

        Returns:

        """
        # 初始化es连接
        self.initEsConnect()
        # 初始化DB ADMIN
        self.initDbAdminConnect()
        # 初始化DB SEARCH
        self.initDbSearchConnect()
        # 初始化 rd_search文件下 的 search_content 段
        self.initRdSearchContentConnect()
        # 初始化 rd_search文件下 的 user_interact_ids 段
        self.initRdUserValidClickAndFavoriteConnect()
        # 初始化 用户分群
        # self.initRdUserProfileConnect()
        # 初始化 es result 缓存
        self.initRdEsResultConnect()

        # 初始化 DB hhz tag
        self.initDbHhzTagConnect()
        # 初始化 DB store
        self.initDbStoreConnect()

    def closeResource(self):
        """
        关闭连接资源
        Args:

        Returns:

        """
        # 关闭es连接
        self.closeEsConnect()
        # 关闭admin 库连接
        self.closeDbAdminConnect()
        # 关闭search 库连接
        self.closeDbSearchConnect()
        # 关闭 redis search content 连接
        self.closeRdSearchContentConnect()
        # 关闭 db hhz tag 连接
        self.closeDbHhzTag()


    def initEsConnect(self):
        """
        初始化es连接
        Args:

        Returns:

        """
        # cls.ES_CONNECT = Elasticsearch(
        #     es_server_ip_port, http_auth=(http_auth_user_name, http_auth_password))
        self.ES_CONNECT = Elasticsearch(es_server_ip_port)
        # Es的连接日志
        # log.logger.debug(self.ES_CONNECT)

    def initRdSearchContentConnect(self):
        """
        初始化redis search content 连接
        Args:

        Returns:

        """
        # redis 读资源
        if host_name == "hzv_ml_gray" and use_gray_redis is True:  # 算法预发机器推荐使用非K8S的redis，我们可以称之为预发ridis
            section = "search_content_ml_gray"
        else:  # 线上K8S和预发K8S，使用的是是K8S的redis，开发环境使用的开发环境的redis，测试环境使用的是测试环境的redis
            section = "search_content"

        self.RD_SEARCH_CONTENT_CONNECTIONPOOL = create_connection.RedisCli.connectionPool(
            root_config["FileLocation"]['rd_search'],
            section=section)

        # redis 写资源
        self.RD_SEARCH_CONTENT_WRITE = create_connection.RedisCli.connect_write(
            root_config["FileLocation"]['rd_search'],
            section=section)

    def initRdSeasonContentConnect(self):
        """
        初始化redis season content 连接
        Args:

        Returns:

        """
        self.RD_SEASON_CONTENT_CONNECTIONPOOL = create_connection.RedisCli.connectionPool(
            root_config["FileLocation"]['rd_search'],
            section="search_content")

    def initRdUserValidClickAndFavoriteConnect(self):
        """
        初始化redis 用户 有效点击历史 和 收藏历史 连接
        Args:

        Returns:

        """
        self.RD_USER_VALID_CLICK_AND_FAVORITE_WRITE = create_connection.RedisCli.connectionPool(
            root_config["FileLocation"]['rd_search'],
            section="user_interact_ids")

    def initRdUserProfileConnect(self):
        """
        初始化redis 用户分群 连接
        Args:

        Returns:

        """
        self.RD_USER_PROFILE_WRITE = create_connection.RedisCli.connectionPool(
            root_config["FileLocation"]['rds_user_profile'],
            section="user_portrait")

    def initRdEsResultConnect(self):
        """
        初始化redis 用户分群 连接
        Args:

        Returns:

        """
        self.RD_ES_RESULT_WRITE = create_connection.RedisCli.connectionPool(
            root_config["FileLocation"]['rd_search'],
            section="es_result")

    def initDbAdminConnect(self):
        """
        初始化db admin 连接
        Args:

        Returns:

        """
        self.DB_ADMIN_CONNECT = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_community'], 'db_admin')

    def initDbHhzTagConnect(self):
        """
        初始化db hhz tag 连接
        Args:

        Returns:

        """
        self.DB_HHZ_TAG_CONNECT = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_tag'], 'db_tag')

    def initDbStoreConnect(self):
        """
        初始化db store 连接
        Args:

        Returns:

        """
        self.DB_STORE_CONNECT = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_store'], 'db_store')

    def initDbSearchConnect(self):
        """
        初始化db search 连接
        Args:

        Returns:

        """
        self.DB_SEARCH_CONNECT = create_connection.MySQLCli.connect(
            root_config["FileLocation"]['db_search'], 'db_search')

    def closeEsConnect(self):
        """
        关闭 es 连接
        Args:

        Returns:

        """
        self.ES_CONNECT.transport.close()
        self.ES_CONNECT = None

    def closeDbAdminConnect(self):
        """
        关闭 db admin 连接
        Args:

        Returns:

        """
        self.DB_ADMIN_CONNECT.close()

    def closeDbSearchConnect(self):
        """
        关闭 db search 连接
        Args:

        Returns:

        """
        self.DB_SEARCH_CONNECT.close()

    def closeRdSearchContentConnect(self):
        """
        关闭 redis search content 连接
        Args:

        Returns:

        """
        self.RD_SEARCH_CONTENT_CONNECTIONPOOL.disconnect()
        self.RD_SEARCH_CONTENT_WRITE.close()

    def closeDbHhzTag(self):
        """
        关闭 db hhz tag 连接
        Args:

        Returns:

        """
        self.DB_HHZ_TAG_CONNECT.close()
