# -*- coding:utf-8 -*-
import os
import random
import numpy as np


class BatchManager:
    #判断数据集问答对是否数量一致，判断是否有训练数据
    def __init__(self, Q, A, batch_size):
        self.Q = Q
        self.A = A
        self.batch_data = []
        self.batch_size = batch_size
        self.make_batch()

    def make_batch(self):
        #断言表达式 如果相等则正常运行，否则打印error信息
        assert len(self.Q) == len(self.A), ValueError("问答数据不一致")
        #zip方法将可迭代对象打成元组最后返回元组list，这里就是将问答组成一对一对，为的是出现相同问句时对应的不同答法，用random来随机
        self.data = list(zip(self.Q, self.A))

        assert len(self.data) > 0, ValueError("训练数据为空")
        sup = len(self.data) % self.batch_size
        sup = 0 if sup == 0 else self.batch_size - sup
        for i in range(sup):
            sup_data = random.choice(self.data)
            self.data.append(sup_data)
        print("-"*50)
        index = 0
        while True:
            if index >= len(self.data):
                break
            data = self.data[index:index+self.batch_size]
            padded_data = self.pad(data)
            index += self.batch_size
            self.batch_data.append(padded_data)
    
    def pad(self, data):
        Q,A = zip(*data)
        Q_max_len = max([len(i) for i in Q])
        A_max_len = max([len(i) for i in A])

        new_Q = []
        new_A = []
        for vec in Q:
            new_vec = vec + [0] * (Q_max_len-len(vec))
            new_Q.append(new_vec)
        
        for vec in A:
            new_vec = vec + [0] * (A_max_len-len(vec))
            new_A.append(new_vec)

        Q = np.array(new_Q)
        A = np.array(new_A)
        return [Q,A]

    def batch(self):
        for i in self.batch_data:
            yield i


#获相对取路径拼接并删除模型和训练集
def clear():
    comfire = input("确认要删除模型重新训练吗？（y/n）: ")
    if comfire in "yY":
        files_under_dir = os.listdir("model")
        for file in files_under_dir:
            try:
                os.remove(os.path.join("model", file))
            except:
                continue
        
        for file in ["A_vocab", "Q_vocab", "map.pkl"]:
            try:
                os.remove(os.path.join("data", file))
            except:
                continue
