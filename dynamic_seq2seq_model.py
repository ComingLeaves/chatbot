# -*- coding:utf-8 -*-
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

class DynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0  

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM 
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
                      控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]

        
    '''
    PAD = 0
    EOS = 2
    UNK = 3
    # 初始化编码解码元件参数 可调。BasicLSTMCell(40),num_units=40表示lstm隐藏层cell数量为40
    def __init__(self,
                encoder_cell=tf.contrib.rnn.BasicLSTMCell(40),
                decoder_cell=tf.contrib.rnn.BasicLSTMCell(40),
                encoder_vocab_size=10,
                decoder_vocab_size=5,
                embedding_size=10,
                attention=True,
                debug=False,
                time_major=False):

        print(encoder_cell)
        self.debug = debug
        self.attention = attention
        self.lstm_dims = 40    # lstm的深度

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        #tf.Variable()为tf的声明变量方法
        self.global_step = tf.Variable(-1, trainable=False)  # 返回所有当前计算中在获取变量时未标记trainable=False的变量集合,即不可训练的参数
        self.max_gradient_norm = 5  # 梯度裁剪

        #创建模型
        self._make_graph()

    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # embedding层
        self._init_embeddings()

        # 双向LSTM并创建encoder
        self._init_bidirectional_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()
    '''
        tf.placeholder(shape,dtype,name) 用于定义过程，返回Tensor类型
            shape是数据形状，默认none，就是一维值，也可以是多维，比如[None,3]表示列是3，行不一定
            dtype是数据类型
            name是名称
        
    '''
    #  初始化占位符
    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.batch_size = tf.shape(self.encoder_inputs)[0]   #input张量或稀疏张量 name：op的名字，用于tensorboard中 out_type默认为tf.int32 返回out_type类型的张量(默认int32)
        # tf.ones函数创建一个shape横为self.batch_size，列为1的所有元素都为1的张量，生成张量中数据类型为int32 该函数返回所有元素都设置为1的张量
        # tf.concat函数用来拼接张量 axis=0或1 axis=0代表在第0个维度拼接 axis=1代表在第1个维度拼接
        # tf.zeros函数用来创建一个张量，所有元素都设为零 这个操作返回一个具有shape形状的dtype类型的张量，所有元素都设置为零 shape代表形状，也就是1维还是2维
        self.decoder_inputs = tf.concat([tf.ones(shape=[self.batch_size, 1], dtype=tf.int32), self.decoder_targets], 1)
        self.decoder_labels = tf.concat([self.decoder_targets, tf.zeros(shape=[self.batch_size, 1], dtype=tf.int32)], 1)
        # tf.abs函数用来求数值的绝对值，可以将多个数值传入list，统一求绝对值
        # tf.sign(x) 根据x大小与0比较 返回-1 0 1
        used = tf.sign(tf.abs(self.encoder_inputs))
        # ft.reduce_sum()计算张量tensor沿某一维度的和，可以在求和后降维  https://www.jianshu.com/p/30b40b504bae
        length = tf.reduce_sum(used, reduction_indices=1)
        # tf.cast()对tensorflow中的张量数据类型转换 这里将length转换成int32类型的
        self.encoder_inputs_length = tf.cast(length, tf.int32)
        used = tf.sign(tf.abs(self.decoder_labels))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.decoder_targets_length = tf.cast(length, tf.int32)

    # _init_embeddings()函数用来初始化嵌入
    def _init_embeddings(self):
        # tf.variable_scope()函数用于定义创建变量(层)的ops的上下文管理器
        with tf.variable_scope("embedding") as scope:
            # math.sqrt()返回一个数的平方根
            sqrt3 = math.sqrt(3)
            # tf.random_uniform_initializer()函数用于生成具有均匀分布的张量的初始化器
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            # encoder Embedding  编码器嵌入
            # tf.get_variable()用来获取已存在的变量，要求名字、初始化等各参数一样，不存在就新建一个
            embedding_encoder = tf.get_variable(
                    "embedding_encoder", 
                    shape=[self.encoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            # tf.nn.embedding_lookup()函数用来选取一个张量里面索引对应的元素
            self.encoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_encoder, self.encoder_inputs
                )
            #  decoder Embedding
            # tf.get_variable()用来获取已存在的变量，要求名字、初始化等各参数一样，不存在就新建一个
            embedding_decoder = tf.get_variable(
                    "embedding_decoder", 
                    shape=[self.decoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            self.embedding_decoder = embedding_decoder
            # tf.nn.embedding_lookup()函数用来选取一个张量里面索引对应的元素
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder, self.decoder_inputs
                )
    #编码
    def _init_bidirectional_encoder(self):
        '''
            双向LSTM encoder
        '''
        # Build RNN cell
        # tf.nn.rnn_cell.BasicLSTMCell()函数用来返回LSTM状态，state_is_tuple默认为True，表示返回的是一个元组，num_units表示神经元的个数，forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息；如果等于0，就都忘记
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dims)
        # tf.nn.dynamic_rnn()函数用来计算RNN输出值Tensor和lstm状态，当前节点状态的输出（state）将成为下个节点的输入。该函数返回的是（outputs,state）元组，然后分别赋值给前边的变量，函数中cell是一个RNNcell实例，inputs:RNN输入，time_major确定tensor的shape顺序，sequence_length当前步长的index超过该巡礼额的实际长度时，该时间步不进行计算，RNN的state复制上一个时间步，同时该时间步的输出全部为零。
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_emb_inp,
            sequence_length=self.encoder_inputs_length, time_major=False,
            dtype=tf.float32
        )
        #得到编码后的中间的上下文向量C，也就是encoder_state
        self.encoder_output = encoder_outputs
        self.encoder_state = encoder_state
    #解码
    def _init_decoder(self):
        # attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
        attention_states = self.encoder_output

        #定义注意力机制
        # tf.contrib.seq2seq.LuongAttention()函数使用解码器的输出值和lstm的深度来定义一个注意力机制，num_units是注意机制权重的size，memory是主体的记忆，就是decoder输出outputs
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.lstm_dims, 
            memory=attention_states,
        )
        # tf.contrib.seq2seq.AttentionWrapper()是一个attention包装器，使用所定义的注意力机制作为解码器单元进行封装。它接受一个RNNcell实例，一个实例注意力机制，一个attention深度参数，返回的输出可以配置为值cell_output而不是attention
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism,
            attention_layer_size=self.lstm_dims
        )
        # Helper
        # Decoder端用来训练的函数，这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target的真实值直接输入给RNN，返回helper对象，可以作为BasicDecoder函数的参数
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_emb_inp, 
            self.decoder_targets_length+1, 
            time_major=False
        )
        '''
        decode_cell输出是每个时间步长的一个hidden_units大小的向量。 
        然而，对于训练和预测，我们需要大小为vocab_size的logits。 
        合理的事情是将线性层（没有激活功能的完全连接层）放在LSTM输出的顶部，以获得非归一化的逻辑。 
        这个层被称为投影层。
        '''
        # tf.layers.Dense()函数表示全连接层，相当于添加一个层，这里用来初始化解码器投影层大小（不确定是不是，待考证）·全连接层，通常在CNN尾部进行重新拟合，用来减少特征信息的损失
        projection_layer = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)
        # 不太理解，是不是先初始化解码器输出值状态，然后再绑定编码器的状态
        init_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)
        # Decoder
        # 该函数生成基本解码器对象(即返回值) cell为RNN层，init_state是RNN的初始状态tensor，output_layer代表输出层， 它是一个tf.layers.Layer的对象
        #这里的训练层使用的是全连接层，默认用的激活函数应该是双曲正切tanh
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=init_state,
            output_layer=projection_layer
        )
        # tf.reduce_max()函数用筛选每行每列的最大值，用axis控制是行筛选还是列筛选
        # tf.round()函数将张量(即tensor)的值四舍五入为最接近的整数，元素，形成相近的张量
        maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length) * 20)
        # Dynamic decoding 动态解码
        # tf.contrib.seq2seq.dynamic_decode()主要作用是接收一个Decoder类，然后依据Encoder进行解码，实现序列的生成(映射)decoder是一个Decoder类，用于解码序列的生成，maximum_iterations最大迭代次数，可以理解为decoder最多可以生成几个词。这个函数主要思想是一步一步的调用Decoder的step函数(该函数接收当前的输入和隐藏层状态会生成下一个词),实现最后一句话的生成。
        outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder, 
            maximum_iterations=maximum_iterations
        )
        self.logits = outputs

        # ------------Infer-----------------
        # Helper
        #GreedyEmbeddingHelper：适用于测试中采用Greedy策略sample的helper
        #tf.fill()创建一个self.batch_size大小的tensor，所有值填充为"1"
        # tf.contrib.seq2seq.GreedyEmbeddingHelper()主要作用是接收开始符，然后生成指定长度大小的句子
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder,
            tf.fill([self.batch_size], 1), 2)
    
        # Decoder 与上面一样，对infer_helper解码
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=infer_helper,
            initial_state=init_state,
            output_layer=projection_layer
            )
        # Dynamic decoding 对基本decoder对象进行动态解码
        infer_outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)
        self.translations = infer_outputs[0][1]

    def _init_optimizer(self):
        # 整理输出并计算loss
        # tf.shape()将一个矩阵的维度，输出成一个维度矩阵，即返回一个一维整数Tensor,用以表示输入input的shape
        # tf.to_float(x)将张量强制转换为float32类型,返回一个形状与x类似的张量或稀疏矩阵
        # tf.sequence_mask()张量变换函数，返回形状为lengths.shape+(maxlen,)的mask张量,投射到指定的dtype，可以作为下面计算loss的权重。lengths是整数张量,其所有值小于等于maxlen。maxlen是标量整数张量,返回张量的最后维度的大小；默认值是lengths中的最大值。
        #current_ts = tf.to_int32(tf.minimum(tf.shape(self.decoder_labels)[1], tf.shape(self.logits)[1]))
        #self.decoder_targets = tf.slice(self.decoder_targets, begin=[0, 0], size=[-1, current_ts])
        #mask = tf.sequence_mask(lengths=self.decoder_targets_length, maxlen=current_ts, dtype=self.logits.dtype)
        #self.logits = tf.slice(self.logits, begin=[0, 0, 0], size=[-1, current_ts, -1])
        #self.decoder_labels = tf.slice(self.decoder_labels, begin=[0, 0], size=[-1, current_ts])

        mask = tf.sequence_mask(
            tf.to_float(self.decoder_targets_length),
            tf.to_float(tf.shape(self.decoder_labels)[1])
        )
        # tf.contrib.seq2seq.sequence_loss()注意会出现logit和label shape不匹配报错，小暖暂时不明原因，网上说要将logits和label shape截断成一样长，网上有操作代码
        # tf.contrib.seq2seq.sequence_loss()对序列logits计算加权交叉熵。training_logits是输出层的结果，targets是目标值，mask在这里作为权重，计算时不会把<PAD>计算进去
        #对self.decoder_labels进行截断
        #对论坛方法做过测试，无效，或许和tensorflow版本有关
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.logits[0][0],
            self.decoder_labels,
            tf.to_float(mask)
        )
        # Calculate and clip gradients 计算和裁剪梯度
        # tf.trainable_variables()用来查看可训练变量，有一些小技巧操作可以百度下
        # tf.gradients(ys,xs)实现ys对xs求导,返回值是一个list，长度等于len(xs)
        #
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        # tf.clip_by_global_norm()是梯度裁剪，目的就是防止梯度爆炸，手段就是控制梯度的最大范式
        clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, self.max_gradient_norm)

        # Optimization 优化
        # tf.train.GradientDescentOptimizer()优化器，学习率为0.1
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        #梯度下降算法
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.train_op = update_step
        # tf.train.Saver()保存和恢复变量
        # tf.global_variables()查看全局变量，以list返回，和tf.trainable_variables()可以加入筛选规则
        self.saver = tf.train.Saver(tf.global_variables())

    def run(self):
        #feed为批量化传递占位符做准备
        feed = {
            self.encoder_inputs: [[2, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            self.decoder_targets: [[1, 1], [1, 1], [4, 1], [3, 1], [2, 0]],
        }
        # tf.Session()是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得要得知的运算结果, 或者是所要运算的部分.
        # tf.global_variables_initializer()返回初始化全局变量op，是variables_initializer(global_variables())的简便写法
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                logits,_,loss = sess.run([self.logits, self.train_op, self.loss], feed_dict=feed)