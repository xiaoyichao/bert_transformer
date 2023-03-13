# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-12-31 13:55:47
LastEditTime: 2022-01-11 18:24:13
Description: 
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")

from common.get_ip import get_host_ip

def send_msg(content):
    """艾特全部，并发送指定信息"""
    import json
    import requests
    # 请求的URL，WebHook地址
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=aa0dd5d6157f3fe86c75530eab1b8accf7025a3e531cd01aa24d7dfc35e15bbe"

    #构建请求头部
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    #构建请求数据
    message = {

        "msgtype": "text",
        "text": {
            "content": get_host_ip()+ ": " +content
        }
        # ,
        # "at": {
        #
        #     "isAtAll": True
        # }

    }
    #对请求的数据进行json封装
    message_json = json.dumps(message)
    #发送请求
    info = requests.post(url=webhook, data=message_json, headers=header)
    #打印返回的结果
    print(info.text)

def send_msg4wechat(content):
    """艾特全部，并发送指定信息"""
    import json, requests
    wx_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a4538755-98b8-4de3-95db-f550f7d9954e"
    # wx_url4ali = "https://oapi.dingtalk.com/robot/send?access_token=aa0dd5d6157f3fe86c75530eab1b8accf7025a3e531cd01aa24d7dfc35e15bbe"
    data = json.dumps({"msgtype": "text", "text": {
                      "content": content, "mentioned_list": ["@all"]}})
    r = requests.post(wx_url, data, auth=('Content-Type', 'application/json'))
