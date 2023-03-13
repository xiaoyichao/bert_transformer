import sys

sys.path.append('')
import presto_con
from presto_con import transaction

HTTP = 'http'
path = '/v1/statement'


class python_presto(object):

    def __init__(self):
        self.scheme = HTTP

    # 定义请求头
    def header(self, user, password):
        headers = {}
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
        headers["Connection"] = "keep-alive"
        headers["X-Presto-Time-Zone"] = "Asia/Shanghai"
        headers["X-Presto-Language"] = "zh-CN,zh;q=0.9"
        headers["x-presto-session"] = "{}={}".format('presto-passwd', user + '##' + password)
        return headers

    def connect(self, host, port, user, password):
        headers = self.header(user, password)

        presto_connection = presto_con.dbapi.connect(
            host=host,
            port=port,
            user=user,
            http_headers=headers,
            isolation_level=transaction.IsolationLevel.AUTOCOMMIT

        )
        return presto_connection
