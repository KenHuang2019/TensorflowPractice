import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)#隨機生成100個數字做為訓練用的input(data base)
y_data = x_data*0.1+0.3 #訓練目標，使下方演算法結構能根據上方數據得出此函數結果

"""
#展示數據庫
print(x_data)
print('數據長度=',len(x_data))
print(y_data)
print('數據長度=',len(y_data))
"""
### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#W可能為多維矩陣，設定變數(Variable)，此處以隨機數練習uniform([維度],數據區間最小值,數據區間最大值)
biases = tf.Variable(tf.zeros([1]))#初始值定義，zero要+s，是複數個"0"

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))#暫時性結果與目標的誤差值
optimizer = tf.train.GradientDescentOptimizer(0.5)#優化器，以此減少誤差，逐步縮小誤差，GradientDescentOptimizer為基礎優化方式，(0.5)為學習效率，一般小於1
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()#初始化變量
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)#重要  要使其活化

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights),sess.run(biases))
