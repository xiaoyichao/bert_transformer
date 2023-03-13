# coding=UTF-8
'''
@Author: xiaoyichao
LastEditors: xiaoyichao
@Date: 2020-04-23 15:52:51
LastEditTime: 2020-09-18 18:23:33
@Description: 
'''

from sanic.response import json


def res_with_head(data_json):
    return json(
        data_json,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,HEAD,GET,POST",
            "Access-Control-Allow-Headers": "x-requested-with"},
        status=200
    )
