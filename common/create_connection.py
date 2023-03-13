import configparser
import redis
import pymysql
import socket
import pysolr
import prestodb
import presto_con as presto_con


class RedisCli:

    @classmethod
    def connect(cls, file_name, section, decoding=False):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = int(cf.get(section, 'rw_separate'))
        except configparser.NoOptionError:
            rw_separate = 0
        if rw_separate:
            cls.host = cf.get(section, 'host').split(',')[1]
            cls.port = cf.get(section, 'port').split(',')[1]
        else:
            cls.host = cf.get(section, 'host')
            cls.port = cf.get(section, 'port')
        cls.password = cf.get(section, 'passwd')
        return redis.StrictRedis(host=socket.gethostbyname(cls.host), port=cls.port, password=cls.password, decode_responses=decoding)

    @classmethod
    def connect_write(cls, file_name, section, decoding=False):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = int(cf.get(section, 'rw_separate'))
        except configparser.NoOptionError:
            rw_separate = 0

        if rw_separate:
            cls.host = cf.get(section, 'host').split(',')[0]
            cls.port = cf.get(section, 'port').split(',')[0]
        else:
            cls.host = cf.get(section, 'host')
            cls.port = cf.get(section, 'port')
        cls.password = cf.get(section, 'passwd')

        return redis.StrictRedis(host=socket.gethostbyname(cls.host), port=cls.port, password=cls.password, decode_responses=decoding)

    @classmethod
    def connectionPool(cls, file_name, section):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = int(cf.get(section, 'rw_separate'))
        except configparser.NoOptionError:
            rw_separate = 0
        if rw_separate:
            cls.host = cf.get(section, 'host').split(',')[1]
            cls.port = cf.get(section, 'port').split(',')[1]
        else:
            cls.host = cf.get(section, 'host')
            cls.port = cf.get(section, 'port')
        cls.password = cf.get(section, 'passwd')
        return redis.ConnectionPool(host=socket.gethostbyname(cls.host), port=cls.port,
                                    password=cls.password, db=0)


# class MySQLCli:
#     @classmethod
#     def connect(cls, file_name, section):
#         cf = configparser.ConfigParser()
#         cf.read(file_name)
#         # 判断是否读写分离
#         try:
#             rw_separate = cf.get(section, 'rw_separate')
#         except configparser.NoOptionError:
#             rw_separate = 0
#         if rw_separate:
#             cls.host = cf.get(section, 'host').split(',')[1]
#         else:
#             cls.host = cf.get(section, 'host')
#         cls.password = cf.get(section, 'passwd')
#         cls.user = cf.get(section, 'user')
#         cls.db = cf.get(section, 'db')
#         return pymysql.connect(host=socket.gethostbyname(cls.host), user=cls.user, password=cls.password, db=cls.db)

class MySQLCli:
    @classmethod
    def connect(cls, file_name, section):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = cf.get(section, 'rw_separate')
        except configparser.NoOptionError:
            rw_separate = 0
        if rw_separate:
            cls.host = cf.get(section, 'host').split(',')[1]
        else:
            cls.host = cf.get(section, 'host')
        cls.password = cf.get(section, 'passwd')
        cls.user = cf.get(section, 'user')
        cls.db = cf.get(section, 'db')
        return pymysql.connect(host=socket.gethostbyname(cls.host), user=cls.user, password=cls.password, db=cls.db, autocommit=True)

    @classmethod
    def connect_write(cls, file_name, section):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = cf.get(section, 'rw_separate')
        except configparser.NoOptionError:
            rw_separate = 0
        if rw_separate:
            cls.host = cf.get(section, 'host').split(',')[0]
        else:
            cls.host = cf.get(section, 'host')
        cls.password = cf.get(section, 'passwd')
        cls.user = cf.get(section, 'user')
        cls.db = cf.get(section, 'db')
        return pymysql.connect(host=socket.gethostbyname(cls.host), user=cls.user, password=cls.password, db=cls.db, autocommit=True)

class SolrCli:
    @classmethod
    def connect(cls, file_name, section):
        cf = configparser.ConfigParser()
        cf.read(file_name)
        # 判断是否读写分离
        try:
            rw_separate = cf.get(section, 'rw_separate')
        except configparser.NoOptionError:
            rw_separate = 0
        if rw_separate:
            host = cf.get(section, 'host').split(',')[1]
        else:
            host = cf.get(section, 'host')
        if not host.startswith("http"):
            host = "http://" + host
        port = cf.get(section, 'port')
        if port == '0':
            port_str = ''
        else:
            port_str = ":" + str(port)
        return pysolr.Solr(host + port_str + "/" + cf.get(section, 'path') + "/")

def presto_execute(sql):
    presto_connection = presto_con.presto_connection.Presto.connect()
    presto_cursor = presto_connection.cursor()
    presto_cursor.execute(sql)
    return presto_cursor.fetchall()

class Presto:
    @classmethod
    def connect(cls):
        return prestodb.dbapi.connect(
            host='172.17.137.242',
            port=8999,
            user='the-user'
        )