'''
Author: root root@haohaozhu.com
Date: 2023-02-03 15:49:08
LastEditors: root root@haohaozhu.com
LastEditTime: 2023-02-03 15:53:59
FilePath: /bert_transformer/bert/embedding_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(5)

width = 768
max_position_embeddings = 512
batch_size,seq_length,num_attention_heads, attention_head_size = 16,128,12,int(width/12)

num_dims = [batch_size,seq_length,width]

def create_initializer(initializer_range=0.02):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
    return initializer

def embedding_lookup(input_ids, vocab_size, embedding_size=128):
    embedding_table = tf.Variable(name="word_embedings")




def get_pos_embedings():
    positions = tf.range(start=0, limit=seq_length, delta=1,dtype=tf.int32)
    print(positions.shape)
    # 这个就是pos embedding的table

    full_position_embeddings = tf.Variable(initial_value=create_initializer(shape=[max_position_embeddings, width]),name="pos_embeddings",trainable=True)
    # 如果不需要那么大的长度，就只保留seq_length的部分
    pos_emb = tf.slice(full_position_embeddings,[0,0], [seq_length, -1])
    print(pos_emb.shape)

    # 二维变三维
    pos_emb = tf.reshape(pos_emb, [1, seq_length, width])





