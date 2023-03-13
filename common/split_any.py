# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao
Date: 2022-01-22 11:23:36
LastEditTime: 2022-01-22 11:28:07
Description: 
'''
import itertools
import math

def split_list(ls, size):
    '''
    Author: xiaoyichao
    param {*}
    Description: 切割list
    '''
    return [ls[i:i+size] for i in range(0, len(ls), size)]

def split_dict(d, n = 1000):
    '''
    Author: xiaoyichao
    param {*}
    Description: 将字典切割，返回[dict,dict,dict]
    '''    
    dict_list = []
    # length of smaller half
    i = iter(d.items()) # alternatively, i = d.iteritems() works in Python 2
    for x in range(math.ceil(len(d)/n)):
        d = dict(itertools.islice(i, n)) # grab first n items
        dict_list.append(d)
    return dict_list

if __name__ == "__main__":
    new_list = split_list([1, 2, 3, 4, 5, 6, 7, 8], 3)
    print(new_list)
