# -*- coding:utf-8 -*-
import jieba
import re
import os
import _pickle as cPickle

class Preprocess():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = {'__PAD__': 0, '__GO__': 1, '__EOS__': 2, '__UNK__': 3}
    def __init__(self):
        self.save_dir = "data"
        self.dialog_dir = "dialog"
        self.Q_vocab = self.vocab.copy()  # 复制vocab字典到Q_vocab中去
        self.A_vocab = self.vocab.copy()
        self.Q_vec = []
        self.A_vec = []

        self.data_map = {}

    def main(self):
        # 加载用户词
        if os.path.isfile(os.path.join(self.save_dir, "word_dict.txt")):
            jieba.load_userdict(os.path.join(self.save_dir, "word_dict.txt"))  # 将自己造的词填入jieba识别

        with open(os.path.join(self.dialog_dir, "Q"), encoding='UTF-8') as Q_file:
            Qs = [i.strip() for i in Q_file.readlines()]   # 列表解析，先执行循环，再i.strip()去掉字符串i的首尾空格
            self.to_vec("Q", Qs)   # 前往to_vec方法

        with open(os.path.join(self.dialog_dir, "A"), encoding='UTF-8') as A_file:
            As = [i.strip() for i in A_file.readlines()]
            self.to_vec("A", As)

        # save 
        self.data_map = {
            "Q_vocab": self.Q_vocab,
            "Q_vec": self.Q_vec,
            "Q_vocab_size": max(self.Q_vocab.values()),
            "A_vocab": self.A_vocab,
            "A_vec": self.A_vec,
            "A_vocab_size": max(self.A_vocab.values()),
        }

        with open(os.path.join(self.save_dir, "map.pkl"), "wb") as f:
            cPickle.dump(self.data_map, f, 0)   # dump将data_map字典写入map.pkl，load从map.pkl中读出字典。注意写入什么类型数据，就读出什么类型数据

    def to_vec(self, dtype, sentences):
        if dtype == "Q":
            vocab = self.Q_vocab
            vec = self.Q_vec
        else:
            vocab = self.A_vocab
            vec = self.A_vec

        max_index = max(vocab.values())   # vocab.values()以列表形式返回字典vocab所有值，然后max取最大赋值
        for sent in sentences:
            segments = jieba.lcut(sent)   # 利用jieba对 问句 分词，返回列表类型
            t_vec = []
            for seg in segments:
                if seg not in vocab:
                    vocab[seg] = max_index + 1  # 添加vocab字典中没有的键，添加键值对
                    max_index += 1
                t_vec.append(vocab.get(seg, 3))  # 获取vocab字典中的seg，如果找不到就返回3，t_vec接收数字3
            if dtype == "A":
                t_vec.append(2)
            vec.append(t_vec)

        # save vocab 
        with open(os.path.join(self.save_dir, dtype+"_vocab"), "w") as f:
            for k,v in vocab.items():   # dict.items以列表返回可遍历的“键值对”元组数组
                f.write("{},{}\n".format(k.encode("utf-8"), v))   # 写入对应文件