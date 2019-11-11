###神經網絡運作時，不可開nicehash，會報錯###

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# add layer
def add_layer(inputs,in_size,out_size,n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):#新增圖層
        with tf.name_scope('Weights'):#新增子圖層
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')#定義權重，隨機產生變量矩陣
            tf.summary.histogram(layer_name+'/Weights',Weights)#可視化為直方圖 tf.summary.histogram(layer_name+'/圖表名稱',資料來源)
        with tf.name_scope('biases'):#新增子圖層
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')#列表
            tf.summary.histogram(layer_name+'/biases',biases)#可視化為直方圖 tf.summary.histogram(layer_name+'/圖表名稱',資料來源)
        with tf.name_scope('Wx_plus_b'):#新增子圖層
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)#可視化為直方圖 tf.summary.histogram(layer_name+'/圖表名稱',資料來源)
        return outputs

# add layer

#data
x_data=np.linspace(-1,1,300)[:, np.newaxis]#創造輸入數據庫
noise=np.random.normal(0, 0.05,x_data.shape)#趨近真實數據用的隨機數
y_data=np.square(x_data)-0.5+noise

#define placeholder for inputs to network
with tf.name_scope('inputs'):#新增圖層
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')#若要利用Tensorboard，需命名，後續結構線會自動生成
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')#若要利用Tensorboard，需命名

#神經元:輸入、輸出層1個、隱藏層10個
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu) #格式(接收data , input_size , output_size , activation_function)
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):#新增圖層
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))#預測與真實結果的誤差
    tf.summary.scalar('loss',loss)#可視化為折線圖，純量在event底下

with tf.name_scope('train'):#新增圖層
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#學習效率，通常小於1，每次練習透過0.X的效率縮小與目標的誤差

init = tf.global_variables_initializer()  #舊版本 tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()#彙整圖表以利下方編寫入for循環內的紀錄
writer = tf.summary.FileWriter("logs/", sess.graph)#新版有更改為tf.summary.FileWriter
sess.run(init)

fig = plt.figure()#生成圖框
ax = fig.add_subplot(1,1,1)#連續性圖像
ax.scatter(x_data,y_data)
plt.ion()
#plt.show()#僅一次性呈現

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})#傳入上面生成的data， feed_dict={x資料來源，y資料來源}
    if i % 50 == 0:#每隔50步紀錄一次
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
