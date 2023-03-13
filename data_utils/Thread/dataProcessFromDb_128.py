# coding=UTF-8
'''
Author: xiaoyichao
LastEditors: xiaoyichao xiaoyichao@haohaozhu.com
Date: 2021-06-10 15:44:34
LastEditTime: 2022-12-20 13:55:06
Description: 
'''
# coding=UTF-8
'''
@Author  : xuzhongjie
@Modify Time  : 2021/5/3 23:34
@Desciption : 获取 info 从db中 
'''
from multiprocessing import  Process

import os,queue
from common.common_log import Logger, get_log_name
from data_utils.data_process import getProcessBlank, getProcessArticleInfo, getProcessNoteInfo, getProcessGuide
from service.Init.InitResource import InitResource
from data_utils.get_cv_vec import get_pic_vec

abs_path = os.path.realpath(__file__)
all_log_name = get_log_name(abs_path, "all")
log = Logger(log_name=all_log_name)

err_log_name = get_log_name(abs_path, "error")
err_log = Logger(log_name=err_log_name)

class dataProcessFromDb(Process):
    '''
    线程类，注意需要继承线程类Thread
    '''
    # 标记线程类
    TYPE = "dataProcessFromDb"

    def __init__(self, thread_id, objIdQueue, data_queue, objNum, content_type, objType, globalTaskNum):
        """
        Args:
            thread_id: 线程id
            queue: 用于存放召回数据存放
            esRecallCls: es召回类 对象
        """
        super(dataProcessFromDb,self).__init__()
        self.thread_id = thread_id
        self.objIdQueue = objIdQueue  # 任务队列
        self.data_queue = data_queue  # es召回对象
        self.objNum = objNum
        self.content_type = content_type
        self.objType = objType
        self.globalTaskNum = globalTaskNum

    def run(self):
        '''
        线程在调用过程中就会调用对应的run方法
        :return:
        '''
        self.recall()

    def recall(self):
        """
        从es召回数据并将数据存入queue中
        Returns:

        """
        try:

            init_resource_cls = InitResource()
            init_resource_cls.initResource()

            while True:
                try:
                    objIds = self.objIdQueue.get(True)
                    if objIds is None:
                        self.data_queue.put(None)
                        break
                except Exception as e:
                    self.data_queue.put(None)
                    break

                if len(objIds) > 0:
                    
                    _, docs_vec, time_consume = get_pic_vec(obj_list=objIds,dim=128)

                    log.logger.info("%s 条数据， cv_time_consume：%s"%(str(len(objIds)),str(time_consume)))
                    id_vec_dict = {}
                    for id, vec in zip(objIds, docs_vec):
                        id_vec_dict[id] = vec

                    # objid_title_remark_dict[blankInfo["bid"]] = {"title": title, "remark": desc, "uid": blankInfo["uid"]}
                    if self.content_type == 1:
                        _, _, _, _, objid_title_remark_dict = getProcessBlank(objIds, init_resource_cls_exterior=init_resource_cls)
                        
                        
                    elif self.content_type == 2:
                        _, _, _, _, objid_title_remark_dict = getProcessNoteInfo(objIds, use_presto=False, init_resource_cls_exterior=init_resource_cls)
                        
                    elif self.content_type == 3:
                        _, _, _, _, objid_title_remark_dict = getProcessArticleInfo(objIds, init_resource_cls_exterior=init_resource_cls)
                        
                    elif self.content_type == 4:
                        _, _, _, _, objid_title_remark_dict = getProcessGuide(objIds)
                    
                    for k, v  in objid_title_remark_dict.items():
                        tmp = v
                        try:
                            cv_vec = id_vec_dict[k]
                        except Exception as e:
                            _, cv_vec, _  = get_pic_vec(obj_list=[k],dim=128)
                            cv_vec = cv_vec[0]

                        tmp["cv_vec"] = cv_vec # 添加cv的特征到value
                        objid_title_remark_dict[k] = tmp
                    self.data_queue.put(objid_title_remark_dict)

                self.globalTaskNum.value = self.globalTaskNum.value - len(objIds)

                print('正在跑的内容类型-' + self.objType, '当前工作的线程为：', self.thread_id, " 剩余数量：", self.globalTaskNum.value)

            init_resource_cls.closeResource()
        except Exception as e:
            self.data_queue.put(None)
            err_log.logger.exception(e)