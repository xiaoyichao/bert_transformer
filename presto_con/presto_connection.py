from __future__ import absolute_import
# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2022-04-19 10:54:22
LastEditTime: 2022-04-19 13:08:44
Description: 
'''
import presto_con


class Presto:
    @classmethod
    def connect(cls):
        # return presto_con.topresto.python_presto().connect(host='172.17.137.242',
        #                                                    port=8999,
        #                                                    user='data_science',
        #                                                    password="DataScience2"
        #                                                    )

        return presto_con.topresto.python_presto().connect(host='172.17.107.241',
                                                           port=18999,
                                                           user='data_science',
                                                           password="DataScience2"
                                                           )

    @classmethod
    def direct_connect(cls):
        return presto_con.dbapi.connect(
            host='172.17.137.242',
            port=8999,
            user='the-user'
        )
    # should only be used if proxy is unavailable
