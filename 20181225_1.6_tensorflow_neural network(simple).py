###神經網絡運作時，不可開nicehash，會報錯###

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# add layer
def add_layer(inputs,in_size,out_size, activation_function=None):
    with tf.name_scope('layer'):#新增圖層
        with tf.name_scope('Weights'):#新增子圖層
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')#定義權重，隨機產生變量矩陣
        with tf.name_scope('biases'):#新增子圖層
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')#列表
        with tf.name_scope('Wx_plus_b'):#新增子圖層
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
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
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu) #格式(接收data , input_size , output_size , activation_function)
prediction = add_layer(l1,10,1,activation_function=None)

with tf.name_scope('loss'):#新增圖層
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))#預測與真實結果的誤差

with tf.name_scope('train'):#新增圖層
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#學習效率，通常小於1，每次練習透過0.X的效率縮小與目標的誤差

init = tf.global_variables_initializer()  #舊版本 tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)#新版有更改為tf.summary.FileWriter
sess.run(init)

fig = plt.figure()#生成圖框
ax = fig.add_subplot(1,1,1)#連續性圖像
ax.scatter(x_data,y_data)
plt.ion()
#plt.show()#僅一次性呈現

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        #print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))#誤差值逐步減少，也就是說逐步將預測值趨近於目的值
        try: #因第一次運算並沒有圖像結果，所以用try讓後續程式可繼續執行，不會馬上報錯
            ax.lines.remove(lines[0])#移除既有暫時性結果，否則會不斷疊加成無法閱讀的圖像
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        
        plt.pause(0.1)

#結果可視化，可更直觀理解


#Visualization tool : Tensorboard
