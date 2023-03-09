# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2021-12-11 15:04:27
LastEditTime: 2021-12-12 13:41:09
Description: 一些普通的功能函数
'''

import os


def get_models(dir_path, reverse=True):
    '''
    @Author: xiaoyichao
    @param {*}
    @Description: 返回文件夹里所有的文件路径（倒序）和最新文件路径
    '''
    model_list = sorted(os.listdir(dir_path), reverse=reverse)
    model_path_list = [os.path.join(dir_path, model_name) for model_name in model_list]
    for mode_path in model_path_list:
        if os.path.isdir(mode_path): # 是否是文件夹
            if not os.listdir(mode_path): # 是否是空文件夹
                model_path_list.remove(mode_path)
                os.rmdir(mode_path )
    newset_model = None
    if len(model_path_list) > 0:
        newset_model = model_path_list[0]
    return model_list, model_path_list, newset_model


def get_models_prefix(dir_path, new_n=0, prefix="", reverse=True):
    '''
    @Author: yangyahe
    @param {*}
    @Description: 返回文件夹里, 特定前缀prefix的, 所有的文件路径（倒序）和第new_n新的文件路径
    '''
    all_file = os.listdir(dir_path)
    if prefix:
        all_file = list(filter(lambda x: x.startswith(prefix), all_file))
    model_list = sorted(all_file, reverse=reverse)
    model_path_list = [os.path.join(dir_path, model_name) for model_name in model_list]
    for mode_path in model_path_list:
        if os.path.isdir(mode_path): # 是否是文件夹
            if not os.listdir(mode_path): # 是否是空文件夹
                model_path_list.remove(mode_path)
                os.rmdir(mode_path )
    newset_model = None
    if len(model_path_list) > new_n:
        newset_model = model_path_list[new_n]
    return model_list, model_path_list, newset_model

def get_models_endfix(dir_path, new_n=0, endfix="", reverse=True):
    '''
    @Author: yangyahe
    @param {*}
    @Description: 返回文件夹里, 特定前缀prefix的, 所有的文件路径（倒序）和第new_n新的文件路径
    '''
    all_file = os.listdir(dir_path)
    if endfix:
        all_file = list(filter(lambda x: x.endswith(endfix), all_file))
    model_list = sorted(all_file, reverse=reverse)
    model_path_list = [os.path.join(dir_path, model_name) for model_name in model_list]
    for mode_path in model_path_list:
        if os.path.isdir(mode_path): # 是否是文件夹
            if not os.listdir(mode_path): # 是否是空文件夹
                model_path_list.remove(mode_path)
                os.rmdir(mode_path )
    newset_model = None
    if len(model_path_list) > new_n:
        newset_model = model_path_list[new_n]
    return model_list, model_path_list, newset_model