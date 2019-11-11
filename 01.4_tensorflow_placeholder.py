import tensorflow as tf
# 類似variable，但在run過程可從外部傳入
input1 = tf.placeholder(tf.float32)#數據形式一般情況下都是float32
input2 = tf.placeholder(tf.float32)
#placeholder用於運算sess.run結果時，才需給定的值
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))#給予字典形式的外部數據
