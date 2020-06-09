# -*- coding:utf-8 -*-
import _pickle as cPickle
import os
import re
import sys
import time

import jieba
import numpy as np
import tensorflow as tf

from action import check_action
from dynamic_seq2seq_model import DynamicSeq2seq
from preprocess import Preprocess
from utils import BatchManager, clear

class Seq2seq():
    '''
    tensorflow-1.0.0

        args:
        encoder_vec_file    encoder向量文件  
        decoder_vec_file    decoder向量文件
        encoder_vocabulary  encoder词典
        decoder_vocabulary  decoder词典
        model_path          模型目录
        batch_size          批处理数
        sample_num          总样本数
        max_batches         最大迭代次数
        show_epoch          保存模型步长
        这里把学习率放在优化器那部分了，不准备采用动态学习率，学习率固定为0.1

    '''
    def __init__(self):
        print("tensorflow version: ", tf.__version__)
        
        self.dict_file = 'data/word_dict.txt'
        self.data_map = "data/map.pkl"    # pkl是cpickle模块生成的文件，用于长久保存字符串、列表、字典等数据

        self.batch_size = 20    # 每次喂进20个
        self.max_epoch = 10000    # 最大100000轮
        self.show_batch = 1     #
        self.model_path = 'model/'
        # jieba导入词典
        jieba.load_userdict(self.dict_file)

        self.location = ["杭州", "重庆", "上海", "北京"]
        self.user_info = {"__UserName__": "yw", "__Location__": "重庆"}
        self.robot_info = {"__RobotName__": "xw"}

        # 获取输入输出
        if os.path.isfile(self.data_map):
            with open(self.data_map, "rb") as f: 
                data_map = cPickle.load(f)  # 使用cpickle读取map.pkl文件内容返回，注意写入是什么类型数据，读取是就是什么类型数据
        else:
            p = Preprocess()
            p.main()    # 如果不存在data_map则调用Preprocess()方法重新创建向量和map
            data_map = p.data_map     # data_map是全局变量的dict，在这里可以取到

        # 从data_map中查找各个键值对并赋值,其中也存在字典嵌套
        self.encoder_vocab = data_map.get("Q_vocab")
        self.encoder_vec = data_map.get("Q_vec")
        self.encoder_vocab_size = data_map.get("Q_vocab_size")
        self.char_to_vec = self.encoder_vocab
        
        self.decoder_vocab = data_map.get("A_vocab")
        self.decoder_vec = data_map.get("A_vec")
        self.decoder_vocab_size = data_map.get("A_vocab_size")
        self.vec_to_char = {v: k for k, v in self.decoder_vocab.items()}

        print("encoder_vocab_size {}".format(self.encoder_vocab_size))
        print("decoder_vocab_size {}".format(self.decoder_vocab_size))
        # 调用DynamicSeq2seq()方法，将编码解码词典长度导入，初始化模型
        self.model = DynamicSeq2seq(
            encoder_vocab_size=self.encoder_vocab_size+1,
            decoder_vocab_size=self.decoder_vocab_size+1,
        )
        #优先给程序分配显存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.restore_model()

    # 恢复模式 通过类方法查找模型，查找不到就初始化全局变量
    def restore_model(self):
        # tf.train.get_checkpoint_state()函数通过checkpoint文件找到模型文件名，该函数返回的是checkpoint文件CheckpointState proto类型的内容
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt:
            print(ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            #初始化，正式给全局变量赋值，这里仅仅只是变量，而不是tensor
            self.sess.run(tf.global_variables_initializer())
            print("没找到模型")

    def get_fd(self, batch, model):
        '''
            获取batch
            为向量填充PAD    
            最大长度为每个batch中句子的最大长度  
            并将数据作转换:  
            [batch_size, time_steps] -> [time_steps, batch_size]
        '''
        encoder_inputs = batch[0]
        decoder_targets = batch[1]
        feed_dict = {
            model.encoder_inputs:encoder_inputs,
            model.decoder_targets:decoder_targets
        }
        return feed_dict
    # 改到这里，暂时转向python web开发
    def train(self):
        print("++++++++train+++++++")
        batch_manager = BatchManager(self.encoder_vec, self.decoder_vec, self.batch_size)

        #用来配置tf的sess，使用gpu还是cpu
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #存放交叉熵结果
        loss_track = []
        total_time = 0
        #算第几轮用的参数
        nums_batch = len(batch_manager.batch_data)
        for epoch in range(self.max_epoch):
            print("[->] epoch {}".format(epoch))
            batch_index = 0
            for batch in batch_manager.batch():
                batch_index += 1
                # 获取fd [time_steps, batch_size]
                fd = self.get_fd(batch, self.model)
                #sess.run计算model的张量tensor，这里利用优化器做优化
                _, loss, logits, labels = self.sess.run([self.model.train_op, 
                                    self.model.loss,
                                    self.model.logits,
                                    self.model.decoder_labels], fd)
                loss_track.append(loss)
                if batch_index % self.show_batch == 0:
                    print("\tstep: {}/{}".format(batch_index, nums_batch))
                    print('\tloss: {}'.format(loss))
                    print("\t"+"-"*50)
                checkpoint_path = self.model_path+"chatbot_seq2seq.ckpt"
                # 保存模型
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        
    def make_inference_fd(self, vec):
        tensor = np.array([vec])
        feed_dict = {
            self.model.encoder_inputs:tensor
        }
        return feed_dict

    def predict(self, input_str):
        segments = jieba.lcut(input_str)
        #xx for x in y 为链表推导式，用于生成链表。xx为链表，for x in y为限定链表的参数，get接收不到返回3，也就是__UN__占位符
        vec = [self.char_to_vec.get(seg, 3) for seg in segments]
        feed = self.make_inference_fd(vec)
        #print(feed)
        logits = self.sess.run([self.model.translations], feed_dict=feed)
        output = logits[0][0].tolist()
        output_str = "".join([self.vec_to_char.get(i, "_UN_") for i in output])
        # check action
        final_output = self.format_output(output_str, input_str)
        #print(final_output)
        return final_output

    #自定义装饰器
    @check_action
    def format_output(self, output_str, raw_input):
        return output_str

    def preprocess(self):
        p = Preprocess()
        p.main()



if __name__ == '__main__':
    # seq = Seq2seq()
    # print("=============")
    # seq.train()

    """
    if sys.argv[1]:
        if sys.argv[1] == 'retrain':
            clear()
            sys.argv[1] = "train"
        seq = Seq2seq()
        if sys.argv[1] == 'train':
            seq.train()
        elif sys.argv[1] == 'infer':
            print("小暖 > "+seq.predict("你叫什么名字"))

    """

    seq = Seq2seq()
    # print("小暖抱着小板凳向这里跑来...")
    time.sleep(1)
    while True:
        Q = input("me >")
        if Q == "exit":
            print("小暖 >小暖要下线喽，拜拜！")
            time.sleep(2)
            print("小暖抱着小板凳跑远了...")
            time.sleep(2)
            break
        print("小暖 > "+seq.predict(Q))

    """
    #clear()
    seq = Seq2seq()
    #seq.predict("天气")
    seq.train()
    """